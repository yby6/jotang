import timm
print(timm.models.create_model('vit_base_patch8_224').default_cfg)

pretrained_cfg_overlay = {'file' : r"pytorch_model (4).bin"}
model = timm.models.create_model('vit_base_patch8_224', pretrained=True, pretrained_cfg_overlay=pretrained_cfg_overlay, num_classes=6)
print(model)