from utilities import *
import torch

def main():
    # Grab the arguments from the CLI
    args = myargs()

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download the data and retrieve the train and test loaders
    train_loader, test_loader = load_data(args.batch_size)

    # Define our model
    model = ConvNet().to(device)

    # Define the Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)

    # Train and Plot the model
    model, loss_list, acc_train, acc_test = training(model, args.num_epochs ,train_loader, test_loader, criterion, optimizer, device)
    plot(loss_list, acc_train, acc_test)

if __name__ == "__main__":
    main()
    