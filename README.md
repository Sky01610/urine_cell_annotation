# urine_cell_annotation
All the file is saved in the content directory. The final output is named as mared_cells.png

For running the code, please run the file as the following sequence:

The segmentation.py is to use the pre-trained U-Net model to segment the urine cell imgae The output image will be restored at "/seg/masks.png".

The extract.py is to extract the single cell image and record the position into a csv file.

The classify.py is to classify the cell type using the single cell image, and record it into the csv file.

The overlay.py is to overlay the original image for final visualization

The annotation file is to add cell type and cell annotation to the overlayed image
