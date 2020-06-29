import torch
import torch.nn as nn
import logging.config
import tqdm

from kvacc.torchutils import collection_to

# Logger
logger = logging.getLogger('kvacc')

class ModelTrainer(object):
    class Listener(object):
        def on_train_begin(self, trainer, params):
            pass

        def on_train_end(self, trainer, params):
            pass

        def on_epoch_begin(self, trainer, params):
            pass

        def on_epoch_end(self, trainer, params):
            pass

        def on_batch_begin(self, trainer, params):
            pass

        def on_batch_end(self, trainer, params):
            pass


    def __init__(self, model=None):
        self.model = model
        self.listeners = []

    def fit(self, train_data_loader=None,
            test_data_loader=None,
            optimizer=None,
            criterion=None,
            n_epochs=100,
            use_cuda=False):

        self.stop_training = False
        device = torch.device("cuda:0" if use_cuda else "cpu")

        self.model.to(device)
        if use_cuda and torch.cuda.device_count() > 1:
            logger.info('Using %d GPUS for BERT' % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        # Callback params
        params = {}
        params['use_cuda'] = use_cuda
        params['device'] = device
        params['model'] = self.model
        params['optimizer'] = optimizer
        params['criterion'] = criterion
        params['n_epochs'] = n_epochs
        params['train.batch_size'] = train_data_loader.batch_size
        params['test.batch_size'] = test_data_loader.batch_size

        logger.info('======================')
        logger.info('Begin training...')
        logger.info('use_cuda, device: %s, %s' % (use_cuda, str(device)))
        logger.info('model: %s' % self.model)
        logger.info('len(train_ds): %s, len(test_ds): %s' % (len(train_data_loader.dataset),
                                                             len(test_data_loader.dataset)))
        # logger.info('X_train[0].shape: %s, y_train[0].shape: %s' % (train_data_loader.dataset[0][0].size(),
        #                                                             train_data_loader.dataset[0][1].size()))
        # logger.info('X_test[0].shape: %s, y_test[0].shape: %s' % (test_data_loader.dataset[0][0].size(),
        #                                                           test_data_loader.dataset[0][1].size()))
        logger.info('optimizer: %s' % optimizer)
        logger.info('criterion: %s' % criterion)
        logger.info('n_epochs: %s' % n_epochs)
        logger.info('train.batch_size: %s' % train_data_loader.batch_size)
        logger.info('test.batch_size: %s' % test_data_loader.batch_size)

        self._fire_train_begin(params)
        for epoch in range(n_epochs):
            if not self.stop_training:
                logger.info('--------------------')
                logger.info('Begin epoch %s/%s' % (epoch, n_epochs))

                params['epoch'] = epoch
                self._fire_epoch_begin(params)

                # Train phase
                logger.info('Begin training phase at epoch %s/%s' % (epoch, n_epochs))

                params['phase'] = 'train'
                self._train_epoch(train_data_loader, params)

                logger.info('End training phase at epoch %s/%s' % (epoch, n_epochs))

                # Validation phase
                logger.info('Begin validation phase at epoch %s/%s' % (epoch, n_epochs))
                params['phase'] = 'val'
                self._train_epoch(test_data_loader, params)
                logger.info('End validation phase at epoch %s/%s' % (epoch, n_epochs))

                self._fire_epoch_end(params)
                logger.info('End epoch %s/%s' % (epoch, n_epochs))
                logger.info('--------------------')

        self._fire_train_end(params)
        logger.info('End training...')
        logger.info('======================')

    def _train_epoch(self, data_loader, params):

        phase = params['phase']
        epoch = params['epoch']
        optimizer = params['optimizer']
        criterion = params['criterion']
        use_cuda = params['use_cuda']
        device = params['device']

        n_data = len(data_loader.dataset)

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%s" % (phase, epoch), total=n_data, bar_format="{l_bar}{r_bar}")
        params['progress_bar'] = data_iter
        for bi, (inputs, targets) in data_iter:
            if hasattr(inputs, '__iter__'):
                inputs = collection_to(inputs, device)
            else:
                inputs = inputs.to(device)

            if hasattr(targets, '__iter__'):
                targets = collection_to(targets, device)
            else:
                targets = targets.to(device)

            params['batch_index'] = bi
            params['inputs'] = inputs
            params['targets'] = targets

            self._fire_batch_begin(params)

            logger.debug('Begin %s batch %s' % (phase, bi))
            logger.debug('inputs: %s' % inputs)
            logger.debug('targets: %s' % targets)

            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            if phase == 'train':
                optimizer.zero_grad()
                # Backpropagation
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

            params['outputs'] = outputs
            params['loss'] = loss

            logger.debug('outputs: %s' % str(outputs))
            logger.debug('loss: %s' % loss)

            self._fire_batch_end(params)

            logger.debug('End %s batch %s' % (phase, bi))

    def add_train_listener(self, listener):
        self.listeners.append(listener)

    def remove_train_listener(self, listener):
        self.listeners.remove(listener)

    def clear_train_listeners(self):
        self.listeners = []

    def _fire_train_begin(self, params):
        for listener in self.listeners:
            listener.on_train_begin(self, params)

    def _fire_train_end(self, params):
        for listener in self.listeners:
            listener.on_train_end(self, params)

    def _fire_epoch_begin(self, params):
        for listener in self.listeners:
            listener.on_epoch_begin(self, params)

    def _fire_epoch_end(self, params):
        for listener in self.listeners:
            listener.on_epoch_end(self, params)

    def _fire_batch_begin(self, params):
        for listener in self.listeners:
            listener.on_batch_begin(self, params)

    def _fire_batch_end(self, params):
        for listener in self.listeners:
            listener.on_batch_end(self, params)

