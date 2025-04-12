import torch

def train(model, train_data, test_data, epochs):
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    print('=========================================')

    model.to(device)

    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    losses = []

    for step in range(epoch):

        for X, y in train_data:
            X = X.to(device)
            y = y.to(device)
            y_logit = model(X)
            loss = criterion(y_logit, y)
            loss.backward()

            optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.95, weight_decay=0.0001)
            optimizer.step()

            optimizer.zero_grad()


        string_one = (f"Epoch {step}: ", loss)

        num_correct = 0
        num_total = 0

        with torch.no_grad():
            model.eval()

            for X, y in test_data:
                X = X.to(device)
                y = y.to(device)

                pred = model(X)
                num_correct = (pred == y).sum().item()
                num_total = X.shape[0]
            
            acc = num_correct / num_total * 100.0
            print(string_one + (", Accuracy: ", acc))

        


        
