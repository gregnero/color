import numpy as np
import cv2
import matplotlib.pyplot as plt
import colorsys

def colorPalette(path, view, max_number_of_colors, hue_separation, sq, vq, space):

    """ Generates a color palette for an image.

    Args:
        path (str): Path to the image.
        view (bool): Choice to view the color palette.
        max_number_of_colors (int): The maximum number of colors you want returned, if possible.
        hue_separation (int): Value to control hue similarity. Higher -> more hue separation. 
        sq (float): [0:1] HSV Saturation quantile that you want represented.
        vq (float): [0:1] HSV Value quantile that you want represented.
        space (str): {'rgb', 'hsv', 'hex'} The color space of the color palette to be returned.
            
    Returns:
        str: Color values of palette given in the specified space.  
    """
   
    #get the title of the image from its path
    title_string = path.split(".")[-2].split("/")[-1]


    #read and convert image
    src_BGR = cv2.imread(path, cv2.IMREAD_COLOR) 
    src_HSV = cv2.cvtColor(src_BGR, cv2.COLOR_BGR2HSV)
    src_RGB = cv2.cvtColor(src_BGR, cv2.COLOR_BGR2RGB)

    #get the rows, cols in the image
    rows = np.shape(src_BGR)[0]
    cols = np.shape(src_BGR)[1]

    #get the channels of interest
    hue_vals = src_HSV[:,:,0]
    sat_vals = src_HSV[:,:,1]
    val_vals = src_HSV[:,:,2] 

    #bin the hue channel at the highest resolution
    number_of_bins = 180
    lower_range_limit = 0
    upper_range_limit = 179
    range_limit = (lower_range_limit, upper_range_limit)
    hue_hist = np.histogram(hue_vals, bins = number_of_bins, range = range_limit)
    hue_distribution = hue_hist[0]

    #sort the hues according to binsize (largest -> smallest)
    ranked_hue_values = np.argsort(-hue_distribution)[:len(hue_distribution)]
    ranked_hue_distribution = hue_distribution[ranked_hue_values]
    
    chosen_hues = []

    for h in range(0, len(ranked_hue_values)):

        if (len(chosen_hues) == max_number_of_colors):

            break
        
        else:

            if (h == 0):

                ranked_hue_value = ranked_hue_values[h]
                previous_ranked_hue_value = ranked_hue_value
                chosen_hues.append(ranked_hue_value)

            else:

                ranked_hue_value = ranked_hue_values[h]

                closeness_bools = np.isclose(chosen_hues, ranked_hue_value, atol = hue_separation)

                if (any(closeness_bools)):

                    continue

                else:

                    chosen_hues.append(ranked_hue_value)

    if (len(chosen_hues) == 0):

        print("ERROR: NO HUES WERE CHOSEN!")
        return (-1)

    saturation_values = []
    value_values = []

    for h in range(0, len(chosen_hues)):

        chosen_hue = chosen_hues[h]

        potential_saturation_values = []
        potential_value_values = []

        for r in range(0,rows):

            for c in range(0,cols):

                hue_val = hue_vals[r,c]
                sat_val = sat_vals[r,c]
                val_val = val_vals[r,c]

                if (chosen_hue == hue_val):

                    potential_saturation_values.append(sat_val)
                    potential_value_values.append(val_val)

        if (len(potential_saturation_values) != 0):

            saturation_values.append(int(np.quantile(potential_saturation_values, sq)))

        if (len(potential_value_values) != 0):

            value_values.append(int(np.quantile(potential_value_values, vq)))

    final_rgb_values = []
    final_hex_values = []
    final_hsv_values = []

    number_of_chosen_hues = len(chosen_hues)
    number_of_saturation_values = len(saturation_values)
    number_of_value_values = len(value_values)
    iter_options = [number_of_chosen_hues, number_of_saturation_values, number_of_value_values]
    my_iter = np.min(iter_options)

    for i in range(0, my_iter):

        h = chosen_hues[i]
        s = saturation_values[i]
        v = value_values[i]

        h_norm = h / upper_range_limit
        s_norm = s / 255
        v_norm = v / 255

        final_hsv_values.append((int(h * 2), np.around(s_norm,2), np.around(v_norm,2)))

        rgb_norm = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)

        r = int(rgb_norm[0] * 255)
        g = int(rgb_norm[1] * 255)
        b = int(rgb_norm[2] * 255)

        rgb = (r, g, b)

        hex_string = '#%02x%02x%02x' % rgb

        final_rgb_values.append(rgb)
        final_hex_values.append(hex_string)

    if (view == True):

        if (my_iter == 1):

            fig, ax = plt.subplots(1,my_iter)

            ax.axis("off")

            blank = np.zeros((500,500,3))

            ax.set_title(final_hex_values[0])

            r = final_rgb_values[0][0]
            g = final_rgb_values[0][1]
            b = final_rgb_values[0][2]
        
            blank[:,:,0] = r / 255
            blank[:,:,1] = g / 255
            blank[:,:,2] = b / 255

            ax.imshow(blank)

        else:

            fig, axs = plt.subplots(1, my_iter)

            iterator = 0
            for ax in axs.reshape(-1):

                ax.axis("off")

                blank = np.zeros((500,500,3))

                ax.set_title(final_hex_values[iterator], fontsize = 7)

                r = final_rgb_values[iterator][0]
                g = final_rgb_values[iterator][1]
                b = final_rgb_values[iterator][2]
            
                blank[:,:,0] = r / 255
                blank[:,:,1] = g / 255
                blank[:,:,2] = b / 255

                ax.imshow(blank)

                iterator = iterator + 1

        plt.show()

    if (space == 'hsv'):
        
        return final_hsv_values
    
    elif (space == 'rgb'):

        return final_rgb_values

    elif (space == 'hex'):

        return final_hex_values

    else:

        print("ERROR: INVALID RETURN SPACE TYPE REQUESTED.")
        print("REQUESTED RETURN SPACE THAT WAS REJECTED:", space)
        return (-1)
