# Some old code...

# #%%capture
#
# def interpolate(tensor_1, tensor_2, num_steps=10):
#   assert tensor_1.shape == tensor_2.shape, "tensors must have same shape"
#   step = (tensor_2 - tensor_1) / num_steps
#   result = [tensor_1 + step * i for i in range(num_steps+1)]
#   assert torch.all(torch.isclose(result[-1], tensor_2, rtol=1e-05, atol=1e-05)), "did not pass interpolate sanity check"
#   return torch.stack(result)
#
# noise1 = torch.randn(nz, 1, 1, device=device)
# noise2 = torch.randn(nz, 1, 1, device=device)
# # noise_step = torch.randn(1, nz, 1, 1, device=device) / 100.0
# # steps = torch.cat([noise + (noise_step * i) for i in range(100)])
# steps = interpolate(noise1, noise2, num_steps=50)
#
# print(steps.shape)
# # steps.shape
#
# with torch.no_grad():
#   image_steps = netG(steps).detach().cpu()
#
# image_steps = (image_steps + 1.0) / 2.0
# image_steps = image_steps.permute(0, 2, 3, 1)
# image_steps.shape
#
# img_list = [image_steps[i] for i in range(len(image_steps))]
#
# fig = plt.figure(figsize=(4, 4))
# plt.axis("off")
# ims = [[plt.imshow(i, animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)
# HTML(ani.to_jshtml())
#
#
# # %%capture
#
# def interpolate(tensor_1, tensor_2, num_steps=10):
#     assert tensor_1.shape == tensor_2.shape, "tensors must have same shape"
#     step = (tensor_2 - tensor_1) / num_steps
#     result = [tensor_1 + step * i for i in range(num_steps + 1)]
#     assert torch.all(
#         torch.isclose(result[-1], tensor_2, rtol=1e-05, atol=1e-05)), "did not pass interpolate sanity check"
#     return torch.stack(result)
#
#
# interpolations = []
# for i in range(4):
#     noise1 = torch.randn(nz, 1, 1, device=device)
#     noise2 = torch.randn(nz, 1, 1, device=device)
#     # noise_step = torch.randn(1, nz, 1, 1, device=device) / 100.0
#     # steps = torch.cat([noise + (noise_step * i) for i in range(100)])
#     steps = interpolate(noise1, noise2, num_steps=50)
#     interpolations.append(steps)
#
# # print(steps.shape)
# # steps.shape
#
# image_interpolations = []å
#
# for latent_interpolation in interpolations:
#     with torch.no_grad():
#         image_steps = netG(latent_interpolation).detach().cpu()
#
#         image_steps = (image_steps + 1.0) / 2.0
#         image_steps = image_steps.permute(0, 2, 3, 1)
#
#         image_interpolations.append(image_steps)
#
# cut_up = []
# for i in range(51):
#     im = torch.zeros((64, 64, 3))
#
#     for j in range(4):
#         r = j // 2
#         c = j % 2
#
#         x1 = int((64 / 2) * c)
#         x2 = int((64 / 2) * (c + 1))
#         y1 = int((64 / 2) * r)
#         y2 = int((64 / 2) * (r + 1))
#
#         # print(image_interpolations[j].shape)
#         # print(x1, x2, y1, y2)
#
#         im[x1:x2, y1:y2, :] = image_interpolations[j][i, x1:x2, y1:y2, :]
#
#     cut_up.append(im)
#
# # plt.imshow(cut_up[0])
#
# fig = plt.figure(figsize=(4, 4))
# plt.axis("off")
# ims = [[plt.imshow(i, animated=True)] for i in cut_up]
# ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)
# HTML(ani.to_jshtml())