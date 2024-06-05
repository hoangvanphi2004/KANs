from data1.dataset import *
from model.kan.kan import *
import torch
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
grid_finer = [5, 10, 20, 50, 100, 200]
#grid_finer = [5, 10]

def create_dataset(f, 
                   n_var=2, 
                   ranges = [-1,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        
        
    train_label = f(train_input)
    test_label = f(test_input)
        
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset

first = 1
old_model = None
learning_rate = 1e-2
train_plot = []
test_plot = []

f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, seed=1002, train_num = 1000, test_num = 1000)

for grid in grid_finer:
    new_model = KAN(G = grid, k = 3, width = [2, 1, 1])
    if(old_model != None):
        new_model.initial_grid_from_other_model(old_model, train_dataloader.dataset[:][0])
    #new_model.plot()
    #optim = torch.optim.Adam(new_model.parameters(), lr = learning_rate)
    optim = torch.optim.LBFGS(new_model.parameters(), lr = 1e-2, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)
    loss_func = torch.nn.MSELoss()
    train_loss_list, test_loss_list = new_model.train_model(train_dataloader, test_dataloader, optimizer = optim, loss_func = loss_func, epochs=20, stop_grid=5, is_LBFGS = True)
    
    # train_plot += train_loss_list
    # test_plot += test_loss_list
    
    if first == 1:
        first = 0
        
    old_model = new_model
    #old_model.plot()
    learning_rate /= 3
    print("#-------------------Evaluate-------------------#")
    print(f"train_loss: {train_loss_list[-1]} test_loss: {test_loss_list[-1]}")

plt.plot([i for i in range(len(train_plot))], train_plot, label="train_loss")
plt.plot([i for i in range(len(test_plot))], test_plot, label="test_loss")
plt.yscale('log', base=10)
plt.legend()
plt.show();


