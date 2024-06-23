from data1.dataset import *
from model.kan.kan import *
import torch
import math
#----------------------------- 2 model -----------------------------------#
# test_kan_model = KAN(G = 3, k = 3, width = [2, 1, 1])
# #optim = torch.optim.LBFGS(test_kan_model.parameters(), lr = 1e-3, history_size= 10, tolerance_grad=1e-32, line_search_fn="strong_wolfe", tolerance_change=1e-32)
# optim = torch.optim.Adam(test_kan_model.parameters(), lr = 5e-3)
# loss_func = RMSE

# test_kan_model.train_model(train_dataloader, test_dataloader, optimizer = optim, loss_func = loss_func, epochs=100, stop_grid=1, is_LBFGS = False)

# test_kan_model.plot()

# temp_model = KAN(G = 3, k = 3, width = [2, 1])

# optim = torch.optim.Adam(temp_model.parameters(), lr = 5e-3)
# loss_func = RMSE

# temp_model.train_model(train_dataloader, test_dataloader, optimizer = optim, loss_func = loss_func, epochs=100, stop_grid=1, is_LBFGS = False)

# temp_model.plot()

# print("Final test\n----------------------")
# print("deeper")
# test_kan_model.test_model(test_dataloader, loss_func=loss_func)
# print("shallower")
# temp_model.test_model(test_dataloader, loss_func=loss_func)

#-----------------------increasing grid----------------------------------#
grid_finer = [5, 10, 20, 50, 100, 200, 500]

first = 1
old_model = None
learning_rate = 5e-2
train_plot = []
test_plot = []

train_data = {}
train_data[0] = train_dataloader.dataset[:][0]
train_data[1] = train_dataloader.dataset[:][1]
test_data = {}
test_data[0] = test_dataloader.dataset[:][0]
test_data[1] = test_dataloader.dataset[:][1]
for grid in grid_finer:
    new_model = KAN(G = grid, k = 3, width = [2, 1, 1])
    if(old_model != None):
        new_model.initial_grid_from_other_model(old_model)
        new_model.plot()
    #optim = torch.optim.Adam(new_model.parameters(), lr = learning_rate)
    optim = torch.optim.LBFGS(new_model.parameters(), lr = 1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
    loss_func = torch.nn.MSELoss()
    train_loss_list, test_loss_list = new_model.train_model(train_data, test_data, optimizer = optim, loss_func = loss_func, epochs=20, stop_grid=5, is_LBFGS = True)
    
    train_plot += train_loss_list
    test_plot += test_loss_list
    
    if first == 1:
        first = 0
    new_model.plot()
    old_model = new_model
    
    learning_rate /= (3 + math.log10(grid) / 2)
    
    print("#-------------------Evaluate-------------------#")
    print(f"train_loss: {train_loss_list[-1]} test_loss: {test_loss_list[-1]}")
    
plt.plot([i for i in range(len(train_plot))], train_plot, label="train_loss")
plt.plot([i for i in range(len(test_plot))], test_plot, label="test_loss")
plt.yscale('log', base=10)
plt.legend()
plt.show();


