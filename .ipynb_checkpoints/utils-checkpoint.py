import torch

def evaluate(network, eval_loader, amp):
    amp &= torch.cuda.is_available()  # Auto disable AMP if CUDA is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    network = network.to(device)
    network.eval()
    total_correct = 0
    
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=amp, dtype=torch.float16):
        for imgs, lbls in eval_loader:
            pred = torch.argmax(network(imgs.to(device)), dim=1).cpu()
            total_correct += (pred == lbls).sum().item()

    network.train()
    return total_correct / len(eval_loader.dataset)

class Step:
    def __init__(self, its_per_ep, it=0):
        self.its_per_ep = its_per_ep
        self.it = it
        
    def update(self):
        self.it += 1
    
    def ep(self):
        return self.it // self.its_per_ep

    def __repr__(self):
        return f"Ep {self.ep()}, it {self.it % self.its_per_ep} ({self.it} total Iterations)"
