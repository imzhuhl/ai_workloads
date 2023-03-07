"""
DeepRec or TensorFlow 1.15

If use TensorFlow 1.15, use --tf flag.

python freeze_test.py --tf -d [cvt_checkpoint_to_savedmodel, freeze_model, ...]
python freeze_test.py --tf -d performance --batch_size 8192
"""

import time
import argparse
import tensorflow as tf
import os
import sys
import math
import collections
from tensorflow.python.client import timeline
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
import json

from tensorflow.python.ops import partitioned_variables

# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.logging.set_verbosity(tf.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))

# Definition of some constants
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ['clicked']
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
HASH_BUCKET_SIZES = {
    'C1': 2500,
    'C2': 2000,
    'C3': 5000000,
    'C4': 1500000,
    'C5': 1000,
    'C6': 100,
    'C7': 20000,
    'C8': 4000,
    'C9': 20,
    'C10': 100000,
    'C11': 10000,
    'C12': 5000000,
    'C13': 40000,
    'C14': 100,
    'C15': 100,
    'C16': 3000000,
    'C17': 50,
    'C18': 10000,
    'C19': 4000,
    'C20': 20,
    'C21': 4000000,
    'C22': 100,
    'C23': 100,
    'C24': 250000,
    'C25': 400,
    'C26': 100000
}


class DLRM():
    def __init__(self,
                 dense_column=None,
                 sparse_column=None,
                 mlp_bot=[512, 256, 64, 16],
                 mlp_top=[512, 256],
                 optimizer_type='adam',
                 learning_rate=0.1,
                 inputs=None,
                 interaction_op='dot',
                 bf16=False,
                 stock_tf=None,
                 adaptive_emb=False,
                 input_layer_partitioner=None,
                 dense_layer_partitioner=None):
        if not inputs:
            raise ValueError('Dataset is not defined.')
        self._feature = inputs[0]
        self._label = inputs[1]

        if not dense_column or not sparse_column:
            raise ValueError('Dense column or sparse column is not defined.')
        self._dense_column = dense_column
        self._sparse_column = sparse_column

        self.tf = stock_tf
        self.bf16 = False if self.tf else bf16
        self.is_training = True
        self._adaptive_emb = adaptive_emb

        self._mlp_bot = mlp_bot
        self._mlp_top = mlp_top
        self._learning_rate = learning_rate
        self._input_layer_partitioner = input_layer_partitioner
        self._dense_layer_partitioner = dense_layer_partitioner
        self._optimizer_type = optimizer_type
        self.interaction_op = interaction_op
        if self.interaction_op not in ['dot', 'cat']:
            print("Invaild interaction op, must be 'dot' or 'cat'.")
            sys.exit()

        self._create_model()
        with tf.name_scope('head'):
            self._create_loss()
            self._create_optimizer()
            self._create_metrics()

    # used to add summary in tensorboard
    def _add_layer_summary(self, value, tag):
        tf.summary.scalar('%s/fraction_of_zero_values' % tag,
                          tf.nn.zero_fraction(value))
        tf.summary.histogram('%s/activation' % tag, value)

    def _dot_op(self, features):
        batch_size = tf.shape(features)[0]
        matrixdot = tf.matmul(features, features, transpose_b=True)
        feature_dim = matrixdot.shape[-1]

        ones_mat = tf.ones_like(matrixdot)
        lower_tri_mat = ones_mat - tf.linalg.band_part(ones_mat, 0, -1)
        lower_tri_mask = tf.cast(lower_tri_mat, tf.bool)
        result = tf.boolean_mask(matrixdot, lower_tri_mask)
        output_dim = feature_dim * (feature_dim - 1) // 2

        return tf.reshape(result, (batch_size, output_dim))

    # create model
    def _create_model(self):
        # input dense feature and embedding of sparse features
        with tf.variable_scope('input_layer', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('dense_input_layer',
                                   partitioner=self._input_layer_partitioner,
                                   reuse=tf.AUTO_REUSE):
                dense_inputs = tf.feature_column.input_layer(
                    self._feature, self._dense_column)
            with tf.variable_scope('sparse_input_layer', reuse=tf.AUTO_REUSE):
                column_tensors = {}
                if self._adaptive_emb and not self.tf:
                    '''Adaptive Embedding Feature Part 1 of 2'''
                    adaptive_mask_tensors = {}
                    for col in CATEGORICAL_COLUMNS:
                        adaptive_mask_tensors[col] = tf.ones([args.batch_size],
                                                             tf.int32)
                    sparse_inputs = tf.feature_column.input_layer(
                        features=self._feature,
                        feature_columns=self._sparse_column,
                        cols_to_output_tensors=column_tensors,
                        adaptive_mask_tensors=adaptive_mask_tensors)
                else:
                    sparse_inputs = tf.feature_column.input_layer(
                        features=self._feature,
                        feature_columns=self._sparse_column,
                        cols_to_output_tensors=column_tensors)

        # MLP behind dense inputs
        mlp_bot_scope = tf.variable_scope(
            'mlp_bot_layer',
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with mlp_bot_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else mlp_bot_scope:
            if self.bf16:
                dense_inputs = tf.cast(dense_inputs, dtype=tf.bfloat16)

            for layer_id, num_hidden_units in enumerate(self._mlp_bot):
                with tf.variable_scope(
                        'mlp_bot_hiddenlayer_%d' % layer_id,
                        reuse=tf.AUTO_REUSE) as mlp_bot_hidden_layer_scope:
                    dense_inputs = tf.layers.dense(
                        dense_inputs,
                        units=num_hidden_units,
                        activation=tf.nn.relu,
                        name=mlp_bot_hidden_layer_scope)
                    dense_inputs = tf.layers.batch_normalization(
                        dense_inputs,
                        training=self.is_training,
                        trainable=True)
                    self._add_layer_summary(dense_inputs,
                                            mlp_bot_hidden_layer_scope.name)
            if self.bf16:
                dense_inputs = tf.cast(dense_inputs, dtype=tf.float32)

        # interaction_op
        if self.interaction_op == 'dot':
            # dot op
            with tf.variable_scope('Op_dot_layer', reuse=tf.AUTO_REUSE):
                mlp_input = [dense_inputs]
                for cols in self._sparse_column:
                    mlp_input.append(column_tensors[cols])
                mlp_input = tf.stack(mlp_input, axis=1)
                mlp_input = self._dot_op(mlp_input)
                mlp_input = tf.concat([dense_inputs, mlp_input], 1)
        elif self.interaction_op == 'cat':
            mlp_input = tf.concat([dense_inputs, sparse_inputs], 1)

        # top MLP before output
        if self.bf16:
            mlp_input = tf.cast(mlp_input, dtype=tf.bfloat16)
        mlp_top_scope = tf.variable_scope(
            'mlp_top_layer',
            partitioner=self._dense_layer_partitioner,
            reuse=tf.AUTO_REUSE)
        with mlp_top_scope.keep_weights(dtype=tf.float32) if self.bf16 \
                else mlp_top_scope:
            for layer_id, num_hidden_units in enumerate(self._mlp_top):
                with tf.variable_scope(
                        'mlp_top_hiddenlayer_%d' % layer_id,
                        reuse=tf.AUTO_REUSE) as mlp_top_hidden_layer_scope:
                    mlp_logits = tf.layers.dense(mlp_input,
                                          units=num_hidden_units,
                                          activation=tf.nn.relu,
                                          name=mlp_top_hidden_layer_scope)

                self._add_layer_summary(mlp_logits, mlp_top_hidden_layer_scope.name)

        if self.bf16:
            mlp_logits = tf.cast(mlp_logits, dtype=tf.float32)

        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE) as logits_scope:
            self._logits = tf.layers.dense(mlp_logits,
                                           units=1,
                                           activation=None,
                                           name=logits_scope)
            self.probability = tf.math.sigmoid(self._logits)
            self.output = tf.round(self.probability)

            self._add_layer_summary(self.probability, logits_scope.name)

    # compute loss

    def _create_loss(self):
        loss_func = tf.keras.losses.BinaryCrossentropy()
        predict = tf.squeeze(self.probability)
        self.loss = tf.math.reduce_mean(loss_func(self._label, predict))
        tf.summary.scalar('loss', self.loss)

    # define optimizer and generate train_op
    def _create_optimizer(self):
        self.global_step = tf.train.get_or_create_global_step()
        if self.tf or self._optimizer_type == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self._learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adamasync':
            optimizer = tf.train.AdamAsyncOptimizer(
                learning_rate=self._learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8)
        elif self._optimizer_type == 'adagraddecay':
            optimizer = tf.train.AdagradDecayOptimizer(
                learning_rate=self._learning_rate,
                global_step=self.global_step)
        elif self._optimizer_type == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate=self._learning_rate,
                initial_accumulator_value=0.1,
                use_locking=False)
        elif self._optimizer_type == 'gradientdescent':
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self._learning_rate)
        else:
            raise ValueError("Optimizer type error.")

        self.train_op = optimizer.minimize(
            self.loss, global_step=self.global_step)

    # compute acc & auc
    def _create_metrics(self):
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self._label,
                                                    predictions=self.output)
        self.auc, self.auc_op = tf.metrics.auc(labels=self._label,
                                               predictions=self.probability,
                                               num_thresholds=1000)
        tf.summary.scalar('eval_acc', self.acc)
        tf.summary.scalar('eval_auc', self.auc)


# generate dataset pipline
def build_model_input(filename, batch_size, num_epochs):
    def parse_csv(value):
        tf.logging.info('Parsing {}'.format(filename))
        cont_defaults = [[0.0] for i in range(1, 14)]
        cate_defaults = [[' '] for i in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cont_defaults + cate_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    '''Work Queue Feature'''
    if args.workqueue and not args.tf:
        from tensorflow.python.ops.work_queue import WorkQueue
        work_queue = WorkQueue([filename])
        # For multiple files：
        # work_queue = WorkQueue([filename, filename1,filename2,filename3])
        files = work_queue.input_dataset()
    else:
        files = filename

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(files)
    # dataset = dataset.shuffle(buffer_size=20000,
    #                           seed=args.seed)  # set seed for reproducing
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(2)
    return dataset


# generate feature columns
def build_feature_columns():
    dense_column = []
    sparse_column = []
    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_COLUMNS:
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                column_name,
                hash_bucket_size=10000,
                dtype=tf.string)

            if not args.tf:
                '''Feature Elimination of EmbeddingVariable Feature'''
                if args.ev_elimination == 'gstep':
                    # Feature elimination based on global steps
                    evict_opt = tf.GlobalStepEvict(steps_to_live=4000)
                elif args.ev_elimination == 'l2':
                    # Feature elimination based on l2 weight
                    evict_opt = tf.L2WeightEvict(l2_weight_threshold=1.0)
                else:
                    evict_opt = None
                '''Feature Filter of EmbeddingVariable Feature'''
                if args.ev_filter == 'cbf':
                    # CBF-based feature filter
                    filter_option = tf.CBFFilter(
                        filter_freq=3,
                        max_element_size=2**30,
                        false_positive_probability=0.01,
                        counter_type=tf.int64)
                elif args.ev_filter == 'counter':
                    # Counter-based feature filter
                    filter_option = tf.CounterFilter(filter_freq=3)
                else:
                    filter_option = None
                ev_opt = tf.EmbeddingVariableOption(
                    evict_option=evict_opt, filter_option=filter_option)

                if args.ev:
                    '''Embedding Variable Feature'''
                    categorical_column = tf.feature_column.categorical_column_with_embedding(
                        column_name, dtype=tf.string, ev_option=ev_opt)
                elif args.adaptive_emb:
                    '''                 Adaptive Embedding Feature Part 2 of 2
                    Expcet the follow code, a dict, 'adaptive_mask_tensors', is need as the input of 
                    'tf.feature_column.input_layer(adaptive_mask_tensors=adaptive_mask_tensors)'.
                    For column 'COL_NAME',the value of adaptive_mask_tensors['$COL_NAME'] is a int32
                    tensor with shape [batch_size].
                    '''
                    categorical_column = tf.feature_column.categorical_column_with_adaptive_embedding(
                        column_name,
                        hash_bucket_size=HASH_BUCKET_SIZES[column_name],
                        dtype=tf.string,
                        ev_option=ev_opt)
                elif args.dynamic_ev:
                    '''Dynamic-dimension Embedding Variable'''
                    print(
                        "Dynamic-dimension Embedding Variable isn't really enabled in model."
                    )
                    sys.exit()

            if args.tf or not args.emb_fusion:
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=16,
                    combiner='mean')
            else:
                '''Embedding Fusion Feature'''
                embedding_column = tf.feature_column.embedding_column(
                    categorical_column,
                    dimension=16,
                    combiner='mean',
                    do_fusion=args.emb_fusion)

            sparse_column.append(embedding_column)
        else:
            column = tf.feature_column.numeric_column(column_name, shape=(1, ))
            dense_column.append(column)

    return dense_column, sparse_column


def train(sess_config,
          input_hooks,
          model,
          data_init_op,
          steps,
          checkpoint_dir,
          tf_config=None,
          server=None):
    model.is_training = True
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), data_init_op),
        saver=tf.train.Saver(max_to_keep=args.keep_checkpoint_max))

    stop_hook = tf.train.StopAtStepHook(last_step=steps)
    log_hook = tf.train.LoggingTensorHook(
        {
            'steps': model.global_step,
            'loss': model.loss
        }, every_n_iter=100)
    hooks.append(stop_hook)
    hooks.append(log_hook)
    if args.timeline > 0:
        hooks.append(
            tf.train.ProfilerHook(save_steps=args.timeline,
                                  output_dir=checkpoint_dir))
    save_steps = args.save_steps if args.save_steps or args.no_eval else steps
    '''
                            Incremental_Checkpoint
    Please add `save_incremental_checkpoint_secs` in 'tf.train.MonitoredTrainingSession'
    it's default to None, Incremental_save checkpoint time in seconds can be set 
    to use incremental checkpoint function, like `tf.train.MonitoredTrainingSession(
        save_incremental_checkpoint_secs=args.incremental_ckpt)`
    '''
    if args.incremental_ckpt and not args.tf:
        print("Incremental_Checkpoint is not really enabled.")
        print("Please see the comments in the code.")
        sys.exit()

    with tf.train.MonitoredTrainingSession(
            master=server.target if server else '',
            is_chief=tf_config['is_chief'] if tf_config else True,
            hooks=hooks,
            scaffold=scaffold,
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_steps=save_steps,
            summary_dir=checkpoint_dir,
            save_summaries_steps=args.save_steps,
            config=sess_config) as sess:
        while not sess.should_stop():
            sess.run([model.loss, model.train_op])
    print("Training completed.")


def eval(sess_config, input_hooks, model, data_init_op, steps, checkpoint_dir):
    model.is_training = False
    hooks = []
    hooks.extend(input_hooks)

    scaffold = tf.train.Scaffold(
        local_init_op=tf.group(tf.local_variables_initializer(), data_init_op))
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=checkpoint_dir, config=sess_config)
    writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'eval'))
    merged = tf.summary.merge_all()

    with tf.train.MonitoredSession(session_creator=session_creator,
                                   hooks=hooks) as sess:
        s = time.time()
        for _in in range(1, steps + 1):
            sess.run([model.acc_op, model.auc_op])
        e = time.time()
        print(f"[TARGET] [dlrm] [Time(sec)]: {e - s}")
        print(f"[TARGET] [dlrm] [Throughput(samples/sec)]: {args.batch_size*steps / (e - s)}")


def main(tf_config=None, server=None):
    # check dataset and count data set size
    print("Checking dataset...")
    train_file = args.data_location + '/eval.csv'
    test_file = args.data_location + '/eval.csv'
    if (not os.path.exists(train_file)) or (not os.path.exists(test_file)):
        print("Dataset does not exist in the given data_location.")
        sys.exit()
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of training dataset is {}".format(no_of_training_examples))
    print("Numbers of test dataset is {}".format(no_of_test_examples))

    # set batch size, eporch & steps
    batch_size = math.ceil(
        args.batch_size / args.micro_batch
    ) if args.micro_batch and not args.tf else args.batch_size

    if args.steps == 0:
        no_of_epochs = 1
        train_steps = math.ceil(
            (float(no_of_epochs) * no_of_training_examples) / batch_size)
    else:
        no_of_epochs = math.ceil(
            (float(batch_size) * args.steps) / no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples) / batch_size)
    print("The training steps is {}".format(train_steps))
    print("The testing steps is {}".format(test_steps))

    # set fixed random seed
    tf.set_random_seed(args.seed)

    # set directory path
    model_dir = os.path.join(args.output_dir,
                             'model_DLRM_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)

    # create data pipline of train & test dataset
    train_dataset = build_model_input(train_file, batch_size, no_of_epochs)
    test_dataset = build_model_input(test_file, batch_size, 1)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    # create feature column
    dense_column, sparse_column = build_feature_columns()

    # create variable partitioner for distributed training
    num_ps_replicas = len(tf_config['ps_hosts']) if tf_config else 0
    input_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.input_layer_partitioner <<
        20) if args.input_layer_partitioner else None
    dense_layer_partitioner = partitioned_variables.min_max_variable_partitioner(
        max_partitions=num_ps_replicas,
        min_slice_size=args.dense_layer_partitioner <<
        10) if args.dense_layer_partitioner else None

    # Session config
    sess_config = tf.ConfigProto()
    if tf_config:
        sess_config.device_filters.append("/job:ps")
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    # Session hooks
    hooks = []

    if args.smartstaged and not args.tf:
        '''Smart staged Feature'''
        next_element = tf.staged(next_element, num_threads=4, capacity=40)
        sess_config.graph_options.optimizer_options.do_smart_stage = True
        hooks.append(tf.make_prefetch_hook())
    if args.op_fusion and not args.tf:
        '''Auto Graph Fusion'''
        sess_config.graph_options.optimizer_options.do_op_fusion = True
    if args.micro_batch and not args.tf:
        '''Auto Mirco Batch'''
        sess_config.graph_options.optimizer_options.micro_batch_num = args.micro_batch

    # create model
    model = DLRM(dense_column=dense_column,
                 sparse_column=sparse_column,
                 learning_rate=args.learning_rate,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=args.tf,
                 adaptive_emb=args.adaptive_emb,
                 interaction_op=args.interaction_op,
                 inputs=next_element,
                 input_layer_partitioner=input_layer_partitioner,
                 dense_layer_partitioner=dense_layer_partitioner)

    # Run model training and evaluation
    train(sess_config, hooks, model, train_init_op, train_steps,
          checkpoint_dir, tf_config, server)
    if not (args.no_eval or tf_config):
        eval(sess_config, hooks, model, test_init_op, test_steps,
             checkpoint_dir)


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'


# Get parse
def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location',
                        help='Full path of train data',
                        required=False,
                        default='.')
    parser.add_argument('--steps',
                        help='set the number of steps on train dataset',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        help='Batch size to train. Default is 512',
                        type=int,
                        default=512)
    parser.add_argument('--output_dir',
                        help='Full path to logs & model output directory',
                        required=False,
                        default='./result')
    parser.add_argument('--checkpoint',
                        help='Full path to checkpoints input/output directory',
                        required=False)
    parser.add_argument('--deep_dropout',
                        help='Dropout regularization for deep model',
                        type=float,
                        default=0.0)
    parser.add_argument('--seed',
                        help='set the random seed for tensorflow',
                        type=int,
                        default=2021)
    parser.add_argument('--optimizer',
                        type=str,
                        choices=['adam', 'adamasync', 'adagraddecay',
                                 'adagrad', 'gradientdescent'],
                        default='adamasync')
    parser.add_argument('--learning_rate',
                        help='Learning rate for model',
                        type=float,
                        default=0.01)
    parser.add_argument('--save_steps',
                        help='set the number of steps on saving checkpoints',
                        type=int,
                        default=0)
    parser.add_argument('--keep_checkpoint_max',
                        help='Maximum number of recent checkpoint to keep',
                        type=int,
                        default=1)
    parser.add_argument('--bf16',
                        help='enable DeepRec BF16 in deep model. Default FP32',
                        action='store_true')
    parser.add_argument('--no_eval',
                        help='not evaluate trained model by eval dataset.',
                        action='store_true')
    parser.add_argument('--timeline',
                        help='number of steps on saving timeline. Default 0',
                        type=int,
                        default=0)
    parser.add_argument('--interaction_op',
                        type=str,
                        choices=['dot', 'cat'],
                        default='cat')
    parser.add_argument('--protocol',
                        type=str,
                        choices=['grpc', 'grpc++', 'star_server'],
                        default='grpc')
    parser.add_argument('--inter',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--intra',
                        help='set inter op parallelism threads.',
                        type=int,
                        default=0)
    parser.add_argument('--input_layer_partitioner',
                        help='slice size of input layer partitioner. units MB',
                        type=int,
                        default=8)
    parser.add_argument('--dense_layer_partitioner',
                        help='slice size of dense layer partitioner. units KB',
                        type=int,
                        default=16)
    parser.add_argument('--tf',
                        help='Use TF 1.15.5 API and disable DeepRec feature to run a baseline.',
                        action='store_true')
    parser.add_argument('--smartstaged',
                        help='Whether to enable smart staged feature of DeepRec, Default to True.',
                        type=boolean_string,
                        default=True)
                        # default=False)
    parser.add_argument('--emb_fusion',
                        help='Whether to enable embedding fusion, Default to True.',
                        type=boolean_string,
                        default=True)
                        # default=False)
    parser.add_argument('--ev',
                        help='Whether to enable DeepRec EmbeddingVariable. Default False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--ev_elimination',
                        help='Feature Elimination of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'l2', 'gstep'],
                        default=None)
    parser.add_argument('--ev_filter',
                        help='Feature Filter of EmbeddingVariable Feature. Default closed.',
                        type=str,
                        choices=[None, 'counter', 'cbf'],
                        default=None)
    parser.add_argument('--op_fusion',
                        help='Whether to enable Auto graph fusion feature. Default to True',
                        type=boolean_string,
                        default=True)
                        # default=False)
    parser.add_argument('--micro_batch',
                        help='Set num for Auto Mirco Batch. Default close.',
                        type=int,
                        default=0) # TODO enable
    parser.add_argument('--adaptive_emb',
                        help='Whether to enable Adaptive Embedding. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('--dynamic_ev',
                        help='Whether to enable Dynamic-dimension Embedding Variable. Default to False.',
                        type=boolean_string,
                        default=False) # TODO enable
    parser.add_argument('--incremental_ckpt',
                        help='Set time of save Incremental Checkpoint. Default 0 to close.',
                        type=int,
                        default=0)
    parser.add_argument('--workqueue',
                        help='Whether to enable Work Queue. Default to False.',
                        type=boolean_string,
                        default=False)
    parser.add_argument('-d', 
                        type=str,
                        choices=[
                            "cvt_checkpoint_to_savedmodel", 
                            "check_checkpoint_model", 
                            "check_load_savedmodel",
                            "freeze_model",
                            "check_frozen_model",
                            "performance"
                        ],
                        default="performance")
    return parser


    '''
    Set some ENV for these DeepRec's features enabled by ENV. 
    More Detail information is shown in https://deeprec.readthedocs.io/zh/latest/index.html.
    START_STATISTIC_STEP & STOP_STATISTIC_STEP: On CPU platform, DeepRec supports memory optimization
        in both stand-alone and distributed trainging. It's default to open, and the 
        default start and stop steps of collection is 1000 and 1100. Reduce the initial 
        cold start time by the following settings.
    MALLOC_CONF: On CPU platform, DeepRec can use memory optimization with the jemalloc library.
        Please preload libjemalloc.so by `LD_PRELOAD=./libjemalloc.so.2 python ...`
    '''
    os.environ['START_STATISTIC_STEP'] = '100'
    os.environ['STOP_STATISTIC_STEP'] = '110'
    os.environ['MALLOC_CONF'] = \
        'background_thread:true,metadata_thp:auto,dirty_decay_ms:20000,muzzy_decay_ms:20000'


def check_checkpoint_model():
    """Load checkpint model, check the predictions
    """
    test_file = args.data_location + '/eval.csv'
    # create data pipline of train & test dataset
    test_dataset = build_model_input(test_file, args.batch_size, 1)

    iterator = tf.data.Iterator.from_structure(test_dataset.output_types,
                                               test_dataset.output_shapes)
    next_element = iterator.get_next()
    test_init_op = iterator.make_initializer(test_dataset)

    # create feature column
    dense_column, sparse_column = build_feature_columns()

    # create model
    model = DLRM(dense_column=dense_column,
                 sparse_column=sparse_column,
                 learning_rate=args.learning_rate,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=args.tf,
                 adaptive_emb=args.adaptive_emb,
                 interaction_op=args.interaction_op,
                 inputs=next_element)
    
    checkpoint_path = "result/model_DLRM_1678180767/model.ckpt-3907"

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        sess.run(test_init_op)
        print(sess.run(model.probability))


def cvt_checkpoint_to_savedmodel():
    # define input placeholder
    inputs = {}
    for x in range(1, 14):
        inputs[f"I{x}"] = tf.placeholder(tf.float32, [None], name=f"I{x}")
    for x in range(1, 27):
        inputs[f"C{x}"] = tf.placeholder(tf.string, [None], name=f"C{x}")
    
    label = tf.placeholder(tf.int32, [None], name="clicked")

    real_input = [inputs, label]

    # create feature column
    dense_column, sparse_column = build_feature_columns()
    
    # create model
    model = DLRM(dense_column=dense_column,
                 sparse_column=sparse_column,
                 learning_rate=args.learning_rate,
                 optimizer_type=args.optimizer,
                 bf16=args.bf16,
                 stock_tf=args.tf,
                 adaptive_emb=args.adaptive_emb,
                 interaction_op=args.interaction_op,
                 inputs=real_input)
    
    checkpoint_path = "result/model_DLRM_1678180767/model.ckpt-3907"

    """
    IteratorGetNext
    make_initializer
    input_layer/dense_input_layer/input_layer/I1/ExpandDims
    input_layer/dense_input_layer/input_layer/concat
    """

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        for node in sess.graph.as_graph_def().node:
            if "ExpandDims" in node.name:
                print(node.name)
                print(node.input)
            # lst = node.input
            # for item in lst:
            #     if "IteratorGetNext" in item:
            #         print(node.name)
            #         print(node.input)
        
        tf.saved_model.simple_save(sess, "./mysaved", inputs=real_input[0], outputs={"output": model.output})


def check_savedmodel():
    """Load savedmodel and check the predictions
    """
    feature_names = [
        "I1", "I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"
    ]

    # read some real data
    test_file = args.data_location + '/eval.csv'
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of test dataset is {}".format(no_of_test_examples))
    f_dict = {}
    with open(test_file) as f:
        for i in range(args.batch_size):
            line = f.readline()
            line = line.strip('\n')
            items = line.split(',')

            if i == 0:
                for name in feature_names:
                    f_dict[f"{name}:0"] =[]

            for i in range(1, 14):
                f_dict[f"I{i}:0"].append(float(items[i]))
            
            for i in range(14, 40):
                f_dict[f"C{i-13}:0"].append(bytes(items[i], 'utf-8'))

    with tf.Session(graph=tf.Graph()) as sess:
        model = tf.saved_model.load(sess, ["serve"], "./mysaved")

        # for node in sess.graph.as_graph_def().node:
        #     lst = node.input
        #     for item in lst:
        #         if "IteratorGetNext" in item:
        #             print(node.name)
        #             print(node.input)

        pred = sess.run(['logits/Sigmoid:0'], feed_dict=f_dict)
        print(pred)


def freeze_model():
    feature_names = [
        "I1", "I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"
    ]
    with tf.Session(graph=tf.Graph()) as sess:
        model = tf.saved_model.load(sess, ["serve"], "./mysaved")

        # Get frozen graph
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), ['logits/Sigmoid'])
        # Save the frozen graph
        tf.io.write_graph(frozen_graph, './frozen_model', 'model.pb', as_text=False)
        tf.io.write_graph(frozen_graph, './frozen_model', 'model.pbtxt', as_text=True)


def check_frozen_model():
    feature_names = [
        "I1", "I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"
    ]

    # every input node's type
    type_enum_list = [tf.float32.as_datatype_enum for i in range(13)]
    type_enum_list.extend([tf.string.as_datatype_enum for i in range(26)])

    # read some real data
    test_file = args.data_location + '/eval.csv'
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of test dataset is {}".format(no_of_test_examples))
    f_dict = {}
    with open(test_file) as f:
        for i in range(args.batch_size):
            line = f.readline()
            line = line.strip('\n')
            items = line.split(',')

            if i == 0:
                for name in feature_names:
                    f_dict[f"{name}:0"] =[]

            for i in range(1, 14):
                f_dict[f"I{i}:0"].append(float(items[i]))
            
            for i in range(14, 40):
                f_dict[f"C{i-13}:0"].append(bytes(items[i], 'utf-8'))


    # read freeze model and optimize for inference
    with tf.gfile.GFile(os.path.join("frozen_model/model.pb"), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_def = optimize_for_inference(
            graph_def,
            feature_names,
            ["logits/Sigmoid"],
            type_enum_list,
            False,
        )
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

        with tf.Session(graph=graph) as sess:
            pred = sess.run(['logits/Sigmoid:0'], feed_dict=f_dict)
            print(pred)


def performance():
    feature_names = [
        "I1", "I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"
    ]

    # every input node's type
    type_enum_list = [tf.float32.as_datatype_enum for i in range(13)]
    type_enum_list.extend([tf.string.as_datatype_enum for i in range(26)])

    # read some real data
    test_file = args.data_location + '/eval.csv'
    no_of_test_examples = sum(1 for line in open(test_file))
    print("Numbers of test dataset is {}".format(no_of_test_examples))
    f_dict = {}
    with open(test_file) as f:
        for i in range(args.batch_size):
            line = f.readline()
            line = line.strip('\n')
            items = line.split(',')

            if i == 0:
                for name in feature_names:
                    f_dict[f"{name}:0"] =[]

            for i in range(1, 14):
                f_dict[f"I{i}:0"].append(float(items[i]))
            
            for i in range(14, 40):
                f_dict[f"C{i-13}:0"].append(bytes(items[i], 'utf-8'))

    sess_config = tf.ConfigProto()
    sess_config.inter_op_parallelism_threads = args.inter
    sess_config.intra_op_parallelism_threads = args.intra

    # read freeze model and optimize for inference
    with tf.gfile.GFile(os.path.join("frozen_model/model.pb"), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_def = optimize_for_inference(
            graph_def,
            feature_names,
            ["logits/Sigmoid"],
            type_enum_list,
            False,
        )
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

        with tf.Session(graph=graph, config=sess_config) as sess: 
            # warmup
            for _ in range(3):
                s = time.time()
                sess.run(["logits/Sigmoid:0"], feed_dict=f_dict)
                e = time.time()
                print(f"Warm Up Time(ms): {(e-s) * 1000}")

            # print(pred)
            time_list = []
            throughput_list = []
            for _ in range(10):
                s = time.time()
                pred = sess.run(['logits/Sigmoid:0'], feed_dict=f_dict)
                e = time.time()
                time_list.append((e - s) * 1000)
                throughput_list.append(args.batch_size / (e - s))
            time_list.sort()
            throughput_list.sort()
            avg_time = sum(time_list) / len(time_list)
            avg_throughput = sum(throughput_list) / len(throughput_list)
            print(f"[TARGET] [dlrm] [Time(ms)]: {avg_time}")
            print(f"[TARGET] [dlrm] [Throughput(samples/sec)]: {avg_throughput}")


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    # cvt_checkpoint_to_savedmodel()
    # check_checkpoint_model()
    # check_load_savedmodel()
    # freeze_model()
    # check_frozen_model()
    # performance()

    exec(f"{args.d}()")
