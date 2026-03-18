import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.ReLU(),
    nn.Linear(4096, 2048)
).to(device)

optimizer = optim.Adam(model.parameters())

data = torch.randn(2048, 2048).to(device)
target = torch.randn(2048, 2048).to(device)

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True
) as prof:

    for step in range(50):

        start = time.time()

        optimizer.zero_grad()

        with record_function("forward"):
            output = model(data)

        loss = (output - target).pow(2).mean()

        with record_function("backward"):
            loss.backward()

        optimizer.step()

        end = time.time()

        print(f"step {step} loss {loss.item():.4f} time {end-start:.4f}")

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))