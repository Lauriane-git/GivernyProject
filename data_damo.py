import dataloader
import show_image

# Specify the path to your folder containing JPEG images
monet_data_path = 'data_jpg/monet_jpg'
image_w, image_h = dataloader.image_cutter_parameters(4)
print(image_w, image_h)
monet_data = dataloader.SmallImageDataset(monet_data_path, 
                dataloader.image_cutter(image_w, image_h))

show_image.show_images(monet_data[:32])