from mrcnn import utils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import time


def judge_duplicate_boxes(combine_bbox,boxes,combine_area,boxes_area):
    ious = utils.compute_iou(combine_bbox,boxes,combine_area,boxes_area)
    return np.all(ious < 1)
    


def combine_boxes(box,boxes,box_area,boxes_area,cover_threshold):
    ious = utils.compute_iou(box,boxes,box_area,boxes_area)
    index = np.where((ious >= cover_threshold) & (ious < 1.00000))
    share_boxes = boxes[index]
    unique_boxes = np.empty((0,4))

    if len(share_boxes) > 0:
        ymin = np.minimum(share_boxes[:,0],box[0])
        xmin = np.minimum(share_boxes[:,1],box[1])
        ymax = np.maximum(share_boxes[:,2],box[2])
        xmax = np.maximum(share_boxes[:,3],box[3])

        combine_boxes = np.array([ymin,xmin,ymax,xmax]).transpose()

        for combine_box in combine_boxes:
            combine_area = (combine_box[3]-combine_box[1]) * (combine_box[2]-combine_box[0])
            if judge_duplicate_boxes(combine_box,boxes,combine_area,boxes_area):
                unique_boxes = np.append(unique_boxes,[combine_box],axis=0)

    return unique_boxes


def rescore(boxes,scores):
    for box in boxes:
        covers   = compute_cover(box,boxes)
        covereds = compute_covered(box,boxes)




    return scores


def split_supession(boxes, scores, cover_threshold):
    while True:
        proposal_count = np.count_nonzero(boxes)
        generate_boxes = np.empty((0,4))
        i=0
        for box in boxes:
            i+=1
            box_area = (box[3] - box[1]) * (box[2] - box[0])
            boxes_area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
            c_boxes = combine_boxes(box,boxes,box_area,boxes_area,cover_threshold)
            generate_boxes = np.append(generate_boxes, c_boxes, axis=0)

        generate_boxes = np.unique(generate_boxes,axis=0)
        print("count of generate_boxes this loop : " + str(len(generate_boxes)))

        num_generate_boxes = len(generate_boxes)
        if num_generate_boxes > 0:
            boxes  = np.append(boxes,generate_boxes,axis=0)
            #scores = rescore(boxes,scores)
            #plot_boxes(boxes,scores,"ss_" + str(i))
            print("last: ")
            print(boxes)
            print("----------")
        else:
            break

    #plot_boxes(boxes,scores,"ss_100")
    return boxes

def plot_boxes(boxes,scores,name, post=False, pre_count = 100):
    fig, ax = plt.subplots()

    ax.set_xlim(0,50)
    ax.set_ylim(0,50)
    colors = ["r", "g", "b", "c", "m", "y", "k", "r", "g", "b", "c", "m", "y", "k", "r", "g", "b", "c", "m", "y", "k"]

    for i, box in enumerate(boxes):
        w = box[3] - box[1]
        h = box[2] - box[0]
        if not post or pre_count > i:
            rect = plt.Rectangle((box[1],box[0]),w,h,fill=False,ec=colors[i])
            ax.text((box[3]+box[1])/2, box[2]+0.5, scores[i], color=colors[i])
            ax.add_patch(rect)
        else:
            rect = plt.Rectangle((box[1]-0.3,box[0]-0.3),w+0.6,h+0.6,fill=False,ec=colors[i])
            ax.text((box[3] + box[1])/2, box[2]+0.5, scores[i], color=colors[i])
            ax.add_patch(rect)

    fig.savefig(name + ".png",format="png",dpi=300)


if __name__ == '__main__':
    boxes =  np.array([[10,10,20,20],[10.5,10.5,12.5,12.5],[15,15,30,30],[5,18,12,25]])
    scores = np.array([0.85,0.99,0.75,0.80,])

    plot_boxes(boxes,scores,"pre_ss")

    t_start = time.time()
    times = 100
    for i in range(times):
        ss_boxes = split_supession(boxes, scores, cover_threshold=0.02)
    t_end= time.time()
    print("per one loop:" + str((t_end - t_start) / times))
