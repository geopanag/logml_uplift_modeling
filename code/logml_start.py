import pandas as pd 
import os
import numpy as np
import torch
from sklearn.model_selection import KFold

from models import BipartiteSAGE2mod
from utils import outcome_regression_loss_l1, uplift_score, set_seed, experiment, evaluate

from torch.optim import Adam

from causalml.inference.meta import BaseXRegressor, BaseTRegressor
from causalml.propensity import ElasticNetPropensityModel
from xgboost import XGBRegressor
from causalml.inference.tree.causal.causaltree import CausalTreeRegressor



def main():
     
    data = torch.load("../data/retail/processed/data.pt")[0]

    results_file_name = "../results/results_start.csv"
    conv_layer = "gcnconv"  # gcnconv, gatconv, sageconv
    
    n_hidden = 16
    lr = 0.01
    l2_reg = 5e-4 
    dropout = 0.2

    no_layers = 1
    out_channels = 1
    num_epochs = 300
    print("Depth of GNN is ", no_layers)

    seed = 0
    # k = 10 # 20,10,7,5,4,3
    validation_fraction = 5
    patience = 50

    print_per_epoch = 50

    criterion = outcome_regression_loss_l1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xp = torch.eye(data['products']['num_products']).to(device)
    xu = data['user']['x'].to(device)
    treatment = data['user']['t'].to(device)
    outcome = data['user']['y'].to(device)
    

    set_seed(seed)
    
    for i in [20,10,7,5,4,3]: # different slip percentage
        k = i
        model_file_name = "../models/"+conv_layer+"_L_"+str(no_layers)+ "_kfold_"+str(k)+ "_model.pt"
        print("training percentage is ", 1/k)
        kf = KFold(n_splits=abs(k), shuffle=True, random_state=seed)

        # make a data frame to gather the results 
        results = [] 

        
        for train_indices, test_indices in kf.split(xu):
            test_indices, train_indices = train_indices, test_indices
            torch.cuda.empty_cache()
            
                
            # split the test indices to test and validation 
            val_indices = train_indices[:int(len(train_indices)/validation_fraction)]
            train_indices = train_indices[int(len(train_indices)/validation_fraction):]


            ## Keep the graph before the treatment and ONLY the edges of the the train nodes (i.e. after the treatment)
            mask = torch.isin(data['user','buys','product']['edge_index'][0, :], torch.tensor(train_indices) )
            edge_index_up_current = data['user','buys','product']['edge_index'][ : , (~data['user','buys','product']['treatment']) | (mask) ]

            edge_index_up_current[1] = edge_index_up_current[1]+ xu.shape[0]

            edge_index_up_current = torch.cat([edge_index_up_current,edge_index_up_current.flip(dims=[0])],dim=1).to(device)

            
            model = BipartiteSAGE2mod(xu.shape[1], xp.shape[1] , n_hidden, out_channels, no_layers, conv_layer, dropout).to(device)
            optimizer = Adam(model.parameters(), lr=lr, weight_decay = l2_reg)

            out = model( xu, xp , edge_index_up_current) # init params

            train_losses, val_losses = experiment(model, optimizer, num_epochs, train_indices, val_indices, edge_index_up_current, treatment, outcome, xu, xp, model_file_name, print_per_epoch, patience,criterion)

            model = torch.load(model_file_name).to(device)
            up40, up20, test_loss = evaluate(model, test_indices, treatment, outcome, xu, xp, edge_index_up_current, criterion)

            # print(f'mse {test_loss:.4f} with avg abs value {torch.mean(torch.abs(outcome[test_indices]))}')
            # print(f'up40 {up40:.4f}')
            # print(f'up20 {up20:.4f}')
            
            result_row = []
            result_row.append(up40)
            result_row.append(up20)
            
        
            # Benchmarks
            train_indices = np.hstack([train_indices, val_indices])

                
            outcome_np = outcome.cpu().numpy()
            xu_np = xu.cpu().numpy()
            treatment_np = treatment.cpu().numpy()

            learner = BaseTRegressor(learner = XGBRegressor())

            learner.fit(X= xu_np[train_indices], y=outcome_np[train_indices], treatment= treatment_np[train_indices] )  
            uplift=learner.predict(X = xu_np[train_indices], treatment = treatment_np[test_indices]).squeeze()

            uplift=learner.predict(X = xu_np[test_indices], treatment= treatment_np[test_indices]).squeeze()

            up40 = uplift_score(uplift, np.hstack([treatment_np[train_indices] ,treatment_np[test_indices]]), np.hstack([outcome_np[train_indices],outcome_np[test_indices]]), rate=0.4)
            up20 = uplift_score(uplift, np.hstack([treatment_np[train_indices],treatment_np[test_indices]]), np.hstack([outcome_np[train_indices],outcome_np[test_indices]]), rate=0.2)

            # print(f'T-learner up40: {up40:.4f} , up20: {up20:.4f}')
            result_row.append(up40)
            result_row.append(up20)

            propensity_model = ElasticNetPropensityModel()

            propensity_model.fit(X=xu_np[train_indices], y = treatment_np[train_indices])
            p_train = propensity_model.predict(X=xu_np[train_indices])
            p_test = propensity_model.predict(X=xu_np[test_indices])

            learner = BaseXRegressor(learner = XGBRegressor())

            learner.fit(X= xu_np[train_indices], y=outcome_np[train_indices], treatment= treatment_np[train_indices], p=p_train )  

            uplift=learner.predict(X = xu_np[test_indices], treatment= treatment_np[test_indices], p=p_test).squeeze()

            up40 = uplift_score(uplift, np.hstack([treatment_np[train_indices] ,treatment_np[test_indices]]), np.hstack([outcome_np[train_indices],outcome_np[test_indices]]), rate=0.4)
            up20 = uplift_score(uplift, np.hstack([treatment_np[train_indices],treatment_np[test_indices]]), np.hstack([outcome_np[train_indices],outcome_np[test_indices]]), rate=0.2)

            # print(f'X-learner up40: {up40:.4f} , up20: {up20:.4f}')
            result_row.append(up40)
            result_row.append(up20)

            
            learner = CausalTreeRegressor(control_name="0")
            X_train = np.hstack(( treatment_np[train_indices].reshape(-1, 1), xu_np[train_indices]))
            X_test = np.hstack((treatment_np[test_indices].reshape(-1, 1),  xu_np[test_indices]))
            learner.fit( X = X_train, treatment = treatment_np[train_indices].astype(str), y=outcome_np[train_indices])
            uplift = learner.predict( X = X_test).squeeze()

            up40 = uplift_score(uplift, np.hstack([treatment_np[train_indices] ,treatment_np[test_indices]]), np.hstack([outcome_np[train_indices],outcome_np[test_indices]]), rate=0.4)
            up20 = uplift_score(uplift, np.hstack([treatment_np[train_indices],treatment_np[test_indices]]), np.hstack([outcome_np[train_indices],outcome_np[test_indices]]), rate=0.2)

            # print(f'Tree up40: {up40:.4f} , up20: {up20:.4f}')
            result_row.append(up40)
            result_row.append(up20)

            
            results.append(result_row)
        
        results = pd.DataFrame(results,columns=['up40_gnn','up20_gnn','up40_t','up20_t','up40_x','up20_x','up40_tree','up20_tree'])
        
        results.to_csv(results_file_name)
        
        print(results.mean())
        print(results.std())


if __name__ == '__main__':
    main()

    
