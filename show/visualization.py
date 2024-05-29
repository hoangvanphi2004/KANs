from model.kan import kan
import matplotlib.pyplot as plt

def plot_loss_scaling_with_epoch(width=[1, 5, 1], k=3, grid_sizes=[3, 5, 10, 20, 50], data_loader=None, test_loader=None, optimizer=None, loss_func=None, epochs=20):
    # Initialize empty lists to store train and test losses
    train_losses = []
    test_losses = []

    # Initialize the KAN model with the first grid size
    model = kan.KAN(width=width, k=k, G=grid_sizes[0])

    # Loop through grid sizes starting from the second one
    for i in range(len(grid_sizes) - 1):
        # Initialize a new KAN model with the current grid size
        model = kan.KAN(width=width, k=k, G=grid_sizes[i])

        # Initialize the grid of the current model from the previous model
        model.initial_grid_from_other_model(model, data_loader.dataset.x[0])

        # Train the model and get the results
        results = model.train(data_loader, test_loader, optimizer, loss_func, epochs)

        # Append train and test losses to the lists
        train_losses += results['train_loss']
        test_losses += results['test_loss']

    # Plot the loss values
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Scaling with Epoch')
    plt.legend()
    plt.show()