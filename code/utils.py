
import torch
import numpy as np


import torch.nn.functional as F
import random 

from typing import Callable
from torch.optim import Optimizer
import random 


def set_seed(seed:int)->None: 
    torch.manual_seed(seed) 
    np.random.seed(seed)
    random.seed(seed)


def train(mask: np.ndarray, 
          model:torch.nn.Module, 
          xu: torch.tensor, 
          xp: torch.tensor, 
          edge_index: torch.tensor, 
          treatment: torch.tensor, 
          outcome: torch.tensor,
          optimizer: Optimizer, 
          criterion: Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor], torch.tensor] ) -> torch.tensor:
    """
    Trains the model for one epoch.
    """
    model.train()
    optimizer.zero_grad() 

    # pred_t, pred_c, hidden_treatment, hidden_control = model(xu, xp, edge_index)    
    # pred_c = model(torch.cat[xu,0], xp, edge_index)
    # loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])
    pred = model(xu, xp, edge_index)
    loss = criterion(treatment[mask], pred[mask], outcome[mask])
    
    loss.backward()  
    optimizer.step() 
    return loss


def test(mask: np.ndarray, 
          model:torch.nn.Module, 
          xu: torch.tensor, 
          xp: torch.tensor, 
          edge_index: torch.tensor, 
          treatment: torch.tensor, 
          outcome: torch.tensor,
          criterion: Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor], torch.tensor] ) -> torch.tensor:
    """
    Tests the model. 
    """
    model.eval()
    # pred_t, pred_c, hidden_treatment, hidden_control = model(xu, xp, edge_index)
    # loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])

    pred = model(torch.cat[xu], xp, edge_index)
    loss = criterion(treatment[mask], pred[mask], outcome[mask])
    return loss


def experiment(model:torch.nn.Module, 
               optimizer : Optimizer, 
               num_epochs: int, 
               train_indices: np.ndarray, 
               val_indices: np.ndarray, 
               edge_index : torch.tensor, 
               treatment: torch.tensor, 
               outcome: torch.tensor, 
               xu: torch.tensor , 
               xp: torch.tensor , 
               model_file: str, 
               print_per_epoch: int, 
               patience: int,
               criterion : Callable[[torch.tensor, torch.tensor, torch.tensor, torch.tensor], torch.tensor]) -> (list,list) :
    """
    Trains the model for num_epochs epochs and returns the train and validation losses.
    """
    early_stopping = 0
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    print_per_epoch = 50
    for epoch in range(num_epochs):
        train_loss = train(train_indices, model, xu, xp, edge_index, treatment, outcome, optimizer, criterion)
        val_loss = test(val_indices, model, xu, xp, edge_index, treatment, outcome, criterion)

        train_losses.append(float(train_loss.item())) 
        val_losses.append(float(val_loss.item()))

        if val_loss < best_val_loss:
            early_stopping=0
            best_val_loss = val_loss
            torch.save(model, model_file)
        else:
            early_stopping += 1
            if early_stopping > patience:
                print("early stopping..")
                break
                
        if epoch%print_per_epoch==0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val loss: {val_loss:.4f}') 
            
    return train_losses, val_losses


def evaluate(model:torch.nn.Module,             
             test_indices: np.ndarray, 
             treatment: torch.tensor, 
             outcome: torch.tensor, 
             xu: torch.tensor, 
             xp: torch.tensor, 
             edge_index : torch.tensor,
             treatment_u:torch.tensor,
             criterion) -> (float, float,float):

    """
    Evaluates the model on the test set.
    """

    model.eval()

    mask = test_indices
    # pred_t, pred_c, hidden_treatment, hidden_control = model(xu, xp, edge_index)
    # test_loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])
    pred_c = model(torch.cat([xu,treatment_u],dim=1), xp, edge_index)
    treatment_u[mask]=1
    pred_t = model(torch.cat([xu,treatment_u],dim=1), xp, edge_index)
    test_loss = criterion(treatment[mask], pred_t[mask], pred_c[mask], outcome[mask])

    treatment_test = treatment[test_indices].detach().cpu().numpy()
    outcome_test = outcome[test_indices].detach().cpu().numpy()
    pred_t = pred_t.detach().cpu().numpy()
    pred_c = pred_c.detach().cpu().numpy()

    uplift = pred_t[test_indices] - pred_c[test_indices]
    uplift = uplift.squeeze()

    up40 = uplift_score(uplift, treatment_test, outcome_test,0.4)
    up20 = uplift_score(uplift, treatment_test, outcome_test,0.2)
    return up40, up20, test_loss


def outcome_regression_loss_l1(t_true: torch.tensor,
                               y_treatment_pred: torch.tensor, 
                               y_control_pred: torch.tensor, 
                               y_true: torch.tensor) -> torch.tensor:
    """
    Compute mse for treatment and control output layers using treatment vector for masking out the counterfactual predictions
    """
    loss0 = torch.mean(((1. - t_true) * F.l1_loss(y_control_pred.squeeze(), y_true, reduction='none')) )
    loss1 = torch.mean((t_true *  F.l1_loss(y_treatment_pred.squeeze(), y_true, reduction='none') ))

    return (loss0 + loss1)/2


def outcome_regression_loss_l1_one_output(y_pred: torch.tensor, 
                               y_true: torch.tensor) -> torch.tensor:
    """
    Compute mse for treatment and control output layers using treatment vector for masking out the counterfactual predictions
    """
    # loss0 = torch.mean(((1. - t_true) * F.l1_loss(y_control_pred.squeeze(), y_true, reduction='none')) )
    loss = F.l1_loss(y_pred.squeeze(), y_true)

    return loss

def outcome_regression_loss(t_true: torch.tensor,
                            y_treatment_pred: torch.tensor, 
                            y_control_pred: torch.tensor, 
                            y_true: torch.tensor) -> torch.tensor:
    """
    Compute mse for treatment and control output layers using treatment vector for masking out the counterfactual predictions
    """
    loss0 = torch.mean((1. - t_true) * F.mse_loss(y_control_pred.squeeze(), y_true, reduction='none')) 
    loss1 = torch.mean(t_true *  F.mse_loss(y_treatment_pred.squeeze(), y_true, reduction='none') )

    return loss0 + loss1


def binary_treatment_loss(t_true, t_pred):
    """
    Compute cross entropy for propensity score , from Dragonnet
    """
    t_pred = (t_pred + 0.001) / 1.002
    
    return torch.mean(F.binary_cross_entropy(t_pred.squeeze(), t_true))




def outcome_regression_loss_dragnn(t_true: torch.tensor,y_treatment_pred: torch.tensor, y_control_pred: torch.tensor, t_pred: torch.tensor, y_true: torch.tensor):
    """
    Compute mse for treatment and control output layers using treatment vector for masking 
    """
   
    loss0 = torch.mean((1. - t_true) * F.mse_loss(y_control_pred.squeeze(), y_true, reduction='none')) 
    loss1 = torch.mean(t_true *  F.mse_loss(y_treatment_pred.squeeze(), y_true, reduction='none') )

    lossT = binary_treatment_loss(t_true.float(), F.sigmoid(t_pred))

    return loss0 + loss1 + lossT




def uplift_score(prediction: torch.tensor, 
                 treatment: torch.tensor, 
                 target: torch.tensor, 
                 rate=0.2) -> float:
    """
    From https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
    Order the samples by the predicted uplift. 
    Calculate the average ground truth outcome of the top rate*100% of the treated and the control samples.
    Subtract the above to get the uplift. 
    """
    order = np.argsort(-prediction)
    treatment_n = int((treatment == 1).sum() * rate)
    treatment_p = target[order][treatment[order] == 1][:treatment_n].mean()

    control_n = int((treatment == 0).sum() * rate)
    control_p = target[order][treatment[order] == 0][:control_n].mean()
    score = treatment_p - control_p
    return score

