startTime = time.time()
for e in range(0, epochs):
    
    # Training
    model.train()

    totalTrainLoss = 0
    totalValLoss = 0

    trainCorrect = 0
    valCorrect = 0

    for (x, y) in trainDataLoader:

        (x, y) = (x.to(device), y.to(device))

        pred = model(x)
        loss = lossFn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
        
    # Evaluation
    with torch.no_grad():

        model.eval()

        for (x, y) in valDataLoader:

            (x, y) = (x.to(device), y.to(device))

            pred = model(x)
            totalValLoss += lossFn(pred, y)

            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Calculating the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # Calculating the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset) # like correcct/total
    valCorrect = valCorrect / len(valDataLoader.dataset)
 
    # Updating training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
 
    print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))
    #tiempo total del entrenamiento
    endTime = time.time()
    print("Total time taken to train the model: {:.2f}s".format(endTime - startTime))
    print("Total time taken per sample: {:.2f}s".format((2138+377)/endTime))
    print("Number of samples in the mini batch: {:.2f}s".format((2138+377)/endTime))
    print("Time taken per batch: {:.2f}s".format(endTime/BATCH_SIZE)
    #print("samples on each batch: {:.2f}s", BATCH_SIZE)
    #BATCH_SIZE