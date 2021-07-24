import copy
import wandb
import numpy as np
from utils.flutils import *
from utils.tools import *
from algorithm.FetchSGD.client import CLIENT
from tqdm import tqdm
from prettytable import PrettyTable


class SERVER:
    def __init__(self, config):
        self.config = config
        self.clients = self.setup_clients()
        self.surrogates = self.setup_surrogates()
        self.clientsTrainSamplesNum = {client.user_id: client.trainSamplesNum for client in self.clients}
        self.clientsTestSamplesNum = {client.user_id: client.testSamplesNum for client in self.clients}
        self.selected_clients = []
        self.losses = []
        self.accs = []
        self.updates = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(algorithm=self.config.algorithm, model_name=self.config.model)
        self.optimal = {
            'round': 0,
            'trainingAcc': -1.0,
            'testAcc': -1.0,
            'trainingLoss': 10e8,
            'testLoss': 10e8,
            'params': None
        }
        self.shape = (self.config.numRows, self.config.numColumns)
        self.Velocity = torch.zeros(self.shape).cuda()
        self.Error = torch.zeros(self.shape).cuda()
        self.gradSize = len(torch.nn.utils.parameters_to_vector(self.model.parameters()))
        use_cuda = config.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.upload_bytes = 0
        self.download_bytes = 0

    @property
    def server_weights(self):
        return get_param_vec(self.model)

    # clients' latest model weights
    @property
    def client_weights(self):
        return {c.user_id: get_param_vec(c.model) for c in self.clients}

    def setup_clients(self):
        users, trainLoaders, testLoaders = setup_datasets(dataset=self.config.dataset,
                                                          batch_size=self.config.batchSize)
        clients = [
            CLIENT(user_id=user_id,
                   trainLoader=trainLoaders[user_id],
                   testLoader=testLoaders[user_id],
                   config=self.config)
            for user_id in users]
        return clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        return np.random.choice(self.clients, self.config.clientsPerRound, replace=False)

    def setup_surrogates(self):
        surrogates = [
            CLIENT(user_id=i,
                   trainLoader=None,
                   testLoader=None,
                   config=self.config)
            for i in range(self.config.clientsPerRound)]
        return surrogates

    def clear(self):
        self.selected_clients = []
        self.losses = []
        self.accs = []
        self.updates = []

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")
        for i in tqdm(range(self.config.numRounds)):
            sumOfSampleNum = 0
            self.selected_clients = self.select_clients(round_th=i)
            for k in range(len(self.selected_clients)):
                surrogate = self.surrogates[k]
                c = self.selected_clients[k]
                new_client_weights = self.get_new_client_weights(c)
                # surrogate <-- c
                surrogate.update(c)
                if str(next(surrogate.model.parameters()).device) == 'cuda:0':
                    surrogate.model.cpu()
                set_param_vec(surrogate.model, new_client_weights)
                trainSamplesNum, update, loss, upload = surrogate.train(round_th=i)
                self.upload_bytes += upload
                self.updates.append((trainSamplesNum, update))
                # c <-- surrogate
                c.update(surrogate)
                # self.client_weights update to the latest after c.update(surrogate)
                sumOfSampleNum += trainSamplesNum
            # print(SKETCH.table)
            SKETCH = args2sketch(grad_size=self.gradSize, num_cols=self.config.numColumns, num_rows=self.config.numRows,
                                 device=torch.device('cuda:0'),
                                 num_blocks=20)
            for (trainSamplesNum, update) in self.updates:
                SKETCH.accumulateTable(trainSamplesNum/sumOfSampleNum * update)

            # weight_update, newVelocity, newError = self.serverSketched(self, sketched_grad=averageUpdate, Velocity=self.Velocity, Error=self.Error, lr=1)
            weight_update = SKETCH.unSketch(k=50000)

            # update global server model
            param_vec = self.server_weights + weight_update.cpu()
            set_param_vec(model=self.model, param_vec=param_vec)

            if i == 0 or (i + 1) % self.config.evalInterval == 0:
                # print(f"\nRound {i}")
                # test on training set
                trainingAccList, trainingLossList = self.test(dataset='train')
                # test on test set
                testAccList, testLossList = self.test(dataset='test')

                # print and log
                self.printAndLog(i, trainingAccList, testAccList, trainingLossList, testLossList)

            self.clear()

    def test(self, dataset='test'):
        accList, lossList = [], []
        surrogate = self.surrogates[0]
        for c in self.clients:
            surrogate.update(c)
            if str(next(surrogate.model.parameters()).device) == 'cuda:0':
                surrogate.model.cpu()
            set_param_vec(surrogate.model, self.server_weights)
            samplesNum, acc, loss = surrogate.test(dataset=dataset)
            accList.append((samplesNum, acc))
            lossList.append((samplesNum, loss))
        return accList, lossList

    def printAndLog(self, round_th, trainingAccList, testAccList, trainingLossList, testLossList):
        trainingAcc = avgMetric(trainingAccList)
        trainingLoss = avgMetric(trainingLossList)
        testAcc = avgMetric(testAccList)
        testLoss = avgMetric(testLossList)

        # post data error, encoder error, trainingAcc. format
        summary = {
            "round": round_th,
            "TrainingAcc": trainingAcc,
            "TestAcc": testAcc,
            "TrainingLoss": trainingLoss,
            "TestLoss": testLoss,
            "UploadBytes": self.upload_bytes,
            "DownloadBytes": self.download_bytes
        }
        wandb.log(summary)

        # table = PrettyTable(['TrainingAcc.', 'TestAcc.', 'TrainingLoss.', 'TestLoss.'])
        #
        # if trainingAcc > self.optimal['trainingAcc']:
        #     self.optimal['trainingAcc'] = trainingAcc
        #     trainingAcc = "\033[1;31m" + f"{round(trainingAcc, 3)}" + "\033[0m"
        # else:
        #     trainingAcc = round(trainingAcc, 3)
        # if testAcc > self.optimal['testAcc']:
        #     self.optimal['testAcc'] = testAcc
        #     testAcc = "\033[1;31m" + f"{round(testAcc, 3)}" + "\033[0m"
        # else:
        #     testAcc = round(testAcc, 3)
        # if trainingLoss < self.optimal['trainingLoss']:
        #     self.optimal['trainingLoss'] = trainingLoss
        #     trainingLoss = "\033[1;31m" + f"{round(trainingLoss, 3)}" + "\033[0m"
        # else:
        #     trainingLoss = round(trainingLoss, 3)
        # if testLoss < self.optimal['testLoss']:
        #     self.optimal['testLoss'] = testLoss
        #     testLoss = "\033[1;31m" + f"{round(testLoss, 3)}" + "\033[0m"
        # else:
        #     testLoss = round(testLoss, 3)
        # table.add_row([trainingAcc, testAcc, trainingLoss, testLoss])
        # print(table)

    def serverSketched(self, sketched_grad, Velocity, Error, lr):
        rho = self.config.globalMomentum
        k = self.config.k
        if self.config.errorType == "local":
            assert self.config.globalMomentum == 0
        elif self.config.errorType == "virtual":
            assert self.config.localMomentum == 0

        torch.add(input=sketched_grad, other=Velocity, alpha=rho, out=Velocity)

        if self.config.errorType == "local":
            Error = Velocity
        elif self.config.errorType == "virtual":
            Error += Velocity
        SKETCH = args2sketch(grad_size=self.gradSize, num_cols=self.config.numColumns, num_rows=self.config.numRows, device=torch.device('cuda:0'),
                             num_blocks=20)
        SKETCH.accumulateTable(Error)
        update = SKETCH.unSketch(k=k)

        # do virtual error
        SKETCH.zero()
        SKETCH.accumulateVec(update)
        sketched_update = SKETCH.table
        if error_type == "virtual":
            nz = sketched_update.nonzero()
            Error[nz[:, 0], nz[:, 1]] = 0

        nz = sketched_update.nonzero()
        Velocity[nz[:, 0], nz[:, 1]] = 0
        return update, Velocity, Error

    def get_new_client_weights(self, client):
        user_id = client.user_id
        local_client_weights = self.client_weights[user_id]
        global_server_weights = self.server_weights
        diff_vec = global_server_weights - local_client_weights
        if self.config.topkDown:
            weight_update = topk(diff_vec, k=self.config.k)
        else:
            weight_update = diff_vec
        download = 4 * torch.ceil(weight_update.abs()).clamp(0, 1).sum()
        self.download_bytes += download
        new_client_weights = local_client_weights + weight_update
        return new_client_weights
