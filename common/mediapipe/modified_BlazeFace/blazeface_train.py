# blazeface_train.py
import torch
import torch.optim as optim
from blazeface import BlazeFace
from blazeface_anchors import AnchorGenerator
from blazeface_loss import MultiBoxLoss
from config import cfg_blazeface_front

# --- 1. Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate Anchors ONCE
generator = AnchorGenerator(cfg_blazeface_front)
priors = generator.forward().to(device) # Shape [896, 4]

# Model
net = BlazeFace(cfg_blazeface_front).to(device)
net.train()

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-4)

# Loss
criterion = MultiBoxLoss(cfg_blazeface_front)

# --- 2. Dummy Data Loader (Replace with Albumentations dataset) ---
# Format: List of Tensors. Each tensor is [Num_Faces, 16] 
# (xmin, ymin, xmax, ymax, kp1x, kp1y... kp6x, kp6y)
def get_dummy_batch(batch_size=8):
    images = torch.randn(batch_size, 3, 128, 128).to(device)
    targets = []
    for _ in range(batch_size):
        # 1 Face per image, center of screen
        # Box: 50, 50, 100, 100
        # Kps: just random points for demo
        face = torch.tensor([[50., 50., 100., 100.] + [75.]*12]).to(device)
        targets.append(face)
    return images, targets

# --- 3. Training Loop ---
print("Starting training...")
for epoch in range(50):
    # Retrieve Batch
    images, targets = get_dummy_batch()
    
    # Zero Grads
    optimizer.zero_grad()
    
    # Forward Pass
    # out returns (conf_preds, loc_preds)
    out = net(images)
    
    # Calculate Loss
    # We pass fixed priors here
    loss_l, loss_c = criterion(out, targets, priors)
    loss = loss_l + loss_c
    
    # Backward
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loc Loss: {loss_l.item():.4f} | Class Loss: {loss_c.item():.4f}")

# Save
torch.save(net.state_dict(), "blazeface_trained.pth")
print("Model saved.")