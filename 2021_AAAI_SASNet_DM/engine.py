import numpy as np
import torch

def evaluate(test_loader, model, device, args):
    model.eval() 
    val_loss = []
    mae = 0.0
    mse = 0.0
    log_para = args.log_para
    for i, (img, target) in enumerate(test_loader):        
        with torch.no_grad():
            img = img.to(device)
            gt_count = target['points'].shape[1]

            pred_map = model(img)
            pred_map = pred_map.data.cpu().numpy()
            pred_cnt = np.sum(pred_map) / log_para

            mae += abs(gt_count - pred_cnt)
            mse += ((gt_count - pred_cnt) * (gt_count - pred_cnt))
            
            if i % args.print_freq == 0:
                print(f'Processing : {target["name"]} image, GT={gt_count:.2f}, Pred={pred_cnt:.2f}, Error={abs(gt_count-pred_cnt):.2f}')

    mae = mae / len(test_loader)
    mse = np.sqrt(mse / len(test_loader))
    
    print('=' * 50)
    print(f'Final Results:')
    print(f'MAE: {mae:.4f}')
    print(f'MSE: {mse:.4f}')
    print('=' * 50)

    return mae, mse