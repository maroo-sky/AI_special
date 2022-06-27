import torch

class ID():
    def __init__(self):
        self.ID = {}
        self.layer_length = 0

    # estimate ID per iteration and update in self.ID dictionary {epoch :{n-th layer: [id_values]}}
    def ID_estimator(self,outputs, epoch):
        # estimate intrinsic dimensionality
        N = outputs[0].size()[0]
        self.layer_length = len(outputs)
        self.ID[str(epoch) + ' epoch'] = {}
        # update dictionary to be included ID in every batch
        for i in range(self.layer_length):
            self.ID[str(epoch) + '_epoch'][str(i) +'_th_layer'] = []

        for i in range(self.layer_length):
            distance = torch.cdist(outputs[i], outputs[i])
            nearest_neighbor = torch.topk(distance, k=3, dim=-1, largest=False)
            r_1 = nearest_neighbor.values.transpose(-1, -2)[1]
            r_2 = nearest_neighbor.values.transpose(-1, -2)[2]
            mu = r_2 / r_1
            mu = torch.log(mu)
            total_mu = torch.sum(mu, dim=-1)

            ID = N / total_mu
            # Update every
            self.ID[str(epoch) + '_epoch'][str(i) +'_th_layer'].append(ID)

    def Global_ID(self):
        for epoch, values in self.layer_length.items():
            for key , value in values.items():
                self.layer_length[epoch][key + '_global_ID'] = sum(value) / len(value)

    def Extract_global_ID(self):
        layer_global_id = {}

        for i in range(self.layer_length):
            layer_global_id[str(i) + '_th_layer_global_ID'] = []

        for epoch, values in self.layer_length.items():
            for key, value in values.items():
                if 'global_ID' in key:
                    layer_global_id[key].append(value)

        return layer_global_id