#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>
#include <unistd.h>

#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static int buff_index = 0;
// static CvCapture * cap;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
double demo_time;
char txt_filename[70];

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes, image buff[3])
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}


void *detect_in_thread(image buff[3], image buff_letter[3])
{
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes, buff);

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    image im = buff[(buff_index+2) % 3];

    FILE *f = fopen(txt_filename, "a");
    if (f == NULL) {
        printf("Error in open txt_filename file\n");
        fclose(f);
        return 0;
    }
    else {

        for(int i = 0; i < nboxes; ++i){
            int class = max_index(dets[i].prob, demo_classes);
            float prob = dets[i].prob[class];
            if(prob > demo_thresh){
                box b = dets[i].bbox;

                int left  = (b.x-b.w/2.)*im.w;
                int right = (b.x+b.w/2.)*im.w;
                int top   = (b.y-b.h/2.)*im.h;
                int bot   = (b.y+b.h/2.)*im.h;

                if(left < 0) left = 0;
                if(right > im.w-1) right = im.w-1;
                if(top < 0) top = 0;
                if(bot > im.h-1) bot = im.h-1;

                fprintf(f, "%d %d %d %d %d %.0f\n", left, right, top, bot, class, prob*100);

            }
        }
        char frame[20] = "-1 -1 -1 -1 -1 -1\n";
        fprintf(f,"%s",frame);
    }

    fclose(f);

    free_detections(dets, nboxes);
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(image buff[3], image buff_letter[3], CvCapture * cap)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void demo(char *cfgfile, char *weightfile, float thresh, const char *filename, char **names, int classes, float hier, int w, int h, int fps)
{
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;

    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);

    int i,sizec = 0;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    CvCapture * cap;
    image *buff;
    image *buff_letter;

    sizec = strlen(filename)-4;
    strncpy(txt_filename,filename,sizec);
    txt_filename[sizec]='\0';
    strcat(txt_filename,".txt");

    printf("video file: %s\n", filename);
    cap = cvCaptureFromFile(filename);
            
    buff = calloc(3, sizeof(image));
    buff_letter = calloc(3, sizeof(image));

    buff[0] = get_image_from_stream(cap);
    if (buff[0].w == 0 && buff[0].h == 0){
        printf("Error broken Video\n");
        return;
    }
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

    while (!demo_done) {
        buff_index = (buff_index + 1) % 3;
        fetch_in_thread(buff, buff_letter, cap);
        detect_in_thread(buff, buff_letter);
    }
    cvReleaseCapture(&cap);

    for(i=0; i<3; i++){
        free(buff[i].data);
        free(buff_letter[i].data);
    }

    free(buff);
    free(buff_letter);

    demo_done = 0;

}
#endif

