import torchvision

vgg16_train = torchvision.models.vgg16(weights = 'DEFAULT') # == pretrained=True
vgg16_untrain = torchvision.models.vgg16(weights = None)    # == pretrained=False

print(vgg16_train)
print(vgg16_untrain)