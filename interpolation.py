import torch

# z1, z2 = torch.randn((1, 100, 1, 1)).to(DEVICE), torch.randn((1, 100, 1, 1)).to(DEVICE)
def interpolate(tensor_1, tensor_2, num_steps=10):
  assert tensor_1.shape == tensor_2.shape, "tensors must have same shape"
  step = (tensor_2 - tensor_1) / num_steps
  result = [tensor_1 + step * i for i in range(num_steps+1)]
  assert torch.all(torch.isclose(result[-1], tensor_2, rtol=1e-05, atol=1e-05)), "did not pass interpolate sanity check"
  return torch.stack(result)

def interpolatation_chain(latent_tensors, num_steps):
  z_from = latent_tensors[0]
  interpolations = []
  for i in range(1, len(latent_tensors)):
    z_to = latent_tensors[i]
    interpolations.append(interpolate(z_from, z_to, num_steps=num_steps))
    z_from = z_to
  return torch.cat(interpolations)