from data1.dataset import *
from model.kan.kan import *
import torch

test_kan_model = KAN(G = 20, k = 3, width = [2, 1, 1])
#optim = torch.optim.LBFGS(test_kan_model.parameters(), lr = 1e-1, history_size= 10, tolerance_grad=1e-32, line_search_fn="strong_wolfe", tolerance_change=1e-32)
optim = torch.optim.Adam(test_kan_model.parameters(), lr = 1e-3)
loss_func = RMSE

epochs = 250
stop_grid = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n -------------------------------");
    
    X = test_kan_model.train_model(train_dataloader, optimizer = optim, loss_func = loss_func, is_LBFGS = False)
    test_kan_model.test_model(test_dataloader, loss_func = loss_func);
    
    if t < stop_grid and t % 2 == 0:
        test_kan_model.update_grid_from_sample(X)
    if t % 30 == 29:
        test_kan_model.plot()
    # if t % 200 == 20:
    #     temp_model = KAN(G = test_kan_model.G * 10, k = 3, width = [2, 1, 1])
        
    #     temp_model.initial_grid_from_other_model(test_kan_model, x = torch.tensor(train_data.dataset[:][0]).to(torch.double))
    #     test_kan_model.plot()
    #     temp_model.plot()
    #     test_kan_model = temp_model
test_kan_model.plot()

