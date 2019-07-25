import torch

from gcommand_loader import GCommandLoader


def predict_test(nn):
    i = 0
    writer = open("test_y", 'w+')
    for instances, tags in test_loader:
        nodes, tags = torch.autograd.Variable(instances), torch.autograd.Variable(tags)
        predictions = nn(nodes)
        predictions = clean_predictions(predictions)
        write_predictions(predictions, list_file_names[i:i + 100], writer)
        i += 100
    writer.close()


def loss_optimizer(net, learning_rate=0.001):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    return loss, optimizer


def clean_predictions(predictions_as_vector):
    predictions = []
    for prediction in predictions_as_vector:
        predictions.append(int(torch.argmax(prediction)))
    return predictions


def train_network(nn, epochs, lr, train_data):
    loss, optimizer = loss_optimizer(nn, lr)
    for epoch in range(epochs):
        for k, data in enumerate(train_data, 0):
            inputs, labels = data
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            optimizer.zero_grad()
            outputs = nn(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
        # check validation set every epoch
        right_prediction = 0
        for examples, tags in valid_loader:
            examples, tags = torch.autograd.Variable(examples), torch.autograd.Variable(tags)
            # forward
            val_outputs = nn(examples)
            val_loss_size = loss(val_outputs, tags)
            right_prediction += val_loss_size.data
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.9
        lr = lr * 0.9
        print("Validation loss = {:.2f}".format(right_prediction / len(valid_loader)))


# write all predictions to test_y
def write_predictions(predictions, file_names, writer):
    i = 0
    for prediction in predictions:
        writer.write(file_names[i])
        writer.write(", ")
        writer.write(str(prediction))
        writer.write('\n')
        i += 1


class CnnModel(torch.nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0)
        self.layer2 = torch.nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=0)
        self.pool_op = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.sizes1 = torch.nn.Linear(1530, 512)
        self.sizes2 = torch.nn.Linear(512, 256)
        self.sizes3 = torch.nn.Linear(256, 64)
        self.sizes4 = torch.nn.Linear(64, 30)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = self.pool_op(x)
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.pool_op(x)
        x = x.view(x.shape[0], -1)
        x = torch.nn.functional.relu(self.sizes1(x))
        x = torch.nn.functional.relu(self.sizes2(x))
        x = torch.nn.functional.relu(self.sizes3(x))
        x = self.sizes4(x)
        return x


# create model instance.
my_model = CnnModel()
dataset = GCommandLoader('train')
validset = GCommandLoader('valid')
testset = GCommandLoader('test')
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=100, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)
valid_loader = torch.utils.data.DataLoader(
    validset, batch_size=100, shuffle=True,
    num_workers=20, pin_memory=True, sampler=None)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=None,
    num_workers=20, pin_memory=True, sampler=None)
list_file_names = []
for el in test_loader.dataset.spects:
    list_file_names.append((el[0].split('/'))[2])
train_network(my_model, 8, 0.001, train_loader)
predict_test(my_model)
