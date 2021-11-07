import PIL.Image as Image
import torchvision.transforms as transform

img = Image.open('H:\\python_workspace\\pytorch\\simple_test_unet\\dataset\\data\\000_HC.png')
transform = transform.ToTensor()
new_img = transform(img)
print(new_img.shape)