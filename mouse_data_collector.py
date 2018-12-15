from pynput import mouse
import csv

Seq_data = []
X_data = []
Y_data = []
##Dataset creation and handling
path_to_csv = r'C:\Users\Jyothi\Desktop\data.csv'
with open(path_to_csv,'w',newline='') as csv_file:
    csv_writer = csv.writer(csv_file,delimiter = ',')
    csv_writer.writerow(['sequence','x_coordinates','y_coordinates'])
    
## Mouse input handling

def on_move(x,y):
    mouse_position_X.append(x)
    mouse_position_Y.append(y)
    
def on_click(x,y,button,pressed):
    return False
    #On clicking the listener exits

##Collecting and storing data
while True:
    mouse_position_X = []
    mouse_position_Y = []
    with mouse.Listener(on_move = on_move, on_click = on_click) as listener:
        listener.join()
    seq_name = input("What sequence is this ?")
    if seq_name == 'q':
        break
    else:
        Seq_data.append(seq_name)
        X_data.append(mouse_position_X)
        Y_data.append(mouse_position_Y)

##Saving to data.csv
with open(path_to_csv,'a',newline='') as csv_file:
    csv_writer = csv.writer(csv_file,delimiter = ',')
    data = zip(Seq_data,X_data,Y_data)
    csv_writer.writerows(list(data))
        
    
    
       
