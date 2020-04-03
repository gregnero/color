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
        space (str): {'rgb', 'bgr', 'hsv', 'hex'} The color space of the color palette to be returned.
            
    Returns:
        list: Color values of palette given in the specified space.  
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

    #bin the hue channel at the highest resolution (opencv hue space goes from 0 -> 179)
    number_of_bins = 180
    lower_range_limit = 0
    upper_range_limit = 179
    range_limit = (lower_range_limit, upper_range_limit)
    hue_hist = np.histogram(hue_vals, bins = number_of_bins, range = range_limit)
    hue_distribution = hue_hist[0]

    #sort the hues according to binsize (largest -> smallest)
    ranked_hue_values = np.argsort(-hue_distribution)[:len(hue_distribution)]
    ranked_hue_distribution = hue_distribution[ranked_hue_values]
   
    #initialize storage for hues to be chosen 
    chosen_hues = []

    #for each of the ranked hues
    for h in range(0, len(ranked_hue_values)):

        #break if max is reached
        if (len(chosen_hues) == max_number_of_colors):

            break
        
        else:

            #always choose top-ranked hue
            if (h == 0):

                ranked_hue_value = ranked_hue_values[h]
                chosen_hues.append(ranked_hue_value)

            
            else:

                ranked_hue_value = ranked_hue_values[h]

                #boolean decision array for closeness evaluation
                closeness_bools = np.isclose(chosen_hues, ranked_hue_value, atol = hue_separation)

                #if any of the booleans are True (ie, if any are within hue_separation)
                if (any(closeness_bools)):

                    continue

                #else, add that sucker
                else:

                    chosen_hues.append(ranked_hue_value)

    #catch exception where no hues are chosen, just in case
    number_of_chosen_hues = len(chosen_hues)
    if (number_of_chosen_hues == 0):

        print("ERROR: NO HUES WERE CHOSEN!")
        return (-1)

    #initialize storage for saturation and value values
    saturation_values = []
    value_values = []

    #for the chosen hues
    for h in range(0, len(chosen_hues)):

        chosen_hue = chosen_hues[h]

        #for this hue, initialize empty potential value arrays
        potential_saturation_values = []
        potential_value_values = []

        #loop through image
        for r in range(0,rows):

            for c in range(0,cols):

                #get hsv value at this point
                hue_val = hue_vals[r,c]
                sat_val = sat_vals[r,c]
                val_val = val_vals[r,c]

                #if the hue value matches up, add the sat, val values cooresponding to it
                if (chosen_hue == hue_val):

                    potential_saturation_values.append(sat_val)
                    potential_value_values.append(val_val)
        
        #if any potental sat, val values were found, choose one according to quantile stats
        if (len(potential_saturation_values) != 0 and len(potential_value_values) != 0):

            saturation_values.append(int(np.quantile(potential_saturation_values, sq)))
            value_values.append(int(np.quantile(potential_value_values, vq)))

    #initialize storage for final values
    final_rgb_values = []
    final_bgr_values = []
    final_hex_values = []
    final_hsv_values = []

    #get minimum iterable number by comparing h,s,v lengths 
    number_of_saturation_values = len(saturation_values)
    number_of_value_values = len(value_values)
    iter_options = [number_of_chosen_hues, number_of_saturation_values, number_of_value_values]
    min_iter = np.min(iter_options)

    for i in range(0, min_iter):
    
        #get hsv
        h = chosen_hues[i]
        s = saturation_values[i]
        v = value_values[i]

        #normalize to allow conversion
        h_norm = h / upper_range_limit
        s_norm = s / 255
        v_norm = v / 255

        #add to final array (nb scaling and rounding)
        final_hsv_values.append((int(h * 2), np.around(s_norm,2), np.around(v_norm,2)))

        #convert back to rgb
        rgb_norm = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)

        #scale to get 'uint8' rgb values
        r = int(rgb_norm[0] * 255)
        g = int(rgb_norm[1] * 255)
        b = int(rgb_norm[2] * 255)

        #create tuple for rgb
        rgb = (r, g, b)
        
        #create tuple for bgr
        bgr = (b, g, r)

        #get hex via rgb
        hex_string = '#%02x%02x%02x' % rgb

        #append the final values for rgb and hex
        final_rgb_values.append(rgb)
        final_bgr_values.append(bgr)
        final_hex_values.append(hex_string)

    #if we want to view it
    if (view == True):

        #special condition for one color
        if (min_iter == 1):

            fig, ax = plt.subplots(1,min_iter)

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

        #special condition for 9 colors (3x3 grid)
        elif (number_of_chosen_hues == 9):

            fig, axs = plt.subplots(3,3)
            plt.subplots_adjust(hspace = 0.4)

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


        #special condition for 4 colors (2x2 grid)
        elif (number_of_chosen_hues == 4):

            fig, axs = plt.subplots(2,2)
            plt.subplots_adjust(hspace = 0.4)

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

        #otherwise, show the available palette
        else:

            fig, axs = plt.subplots(1, min_iter)

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

    #return information to user
    if (space == 'hsv'):
        
        return final_hsv_values
    
    elif (space == 'rgb'):

        return final_rgb_values

    elif (space == 'bgr'):

        return final_bgr_values

    elif (space == 'hex'):

        return final_hex_values

    else:

        print("ERROR: INVALID RETURN SPACE TYPE REQUESTED.")
        print("REQUESTED RETURN SPACE THAT WAS REJECTED:", space)
        return (-1)
