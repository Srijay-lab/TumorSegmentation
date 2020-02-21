import matplotlib.pyplot as plt
import math
import numpy as np

#Print first few figures from the image set
def show_figures(images_data_generator,class_names,k):
    plt.figure(figsize=(10, 10))
    grid_scale = math.ceil(math.sqrt(k))
    x, y = images_data_generator.next()
    y=np.argmax(y,axis=1)
    print(y)
    print(images_data_generator.class_indices)
    for i in range(k):
        plt.subplot(grid_scale, grid_scale, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i]) #cmap=plt.cm.binary - include it as an another argument if you want to check binary map
        plt.xlabel(class_names[y[i]])
    plt.show()

def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')