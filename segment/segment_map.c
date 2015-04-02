#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static int parse_date(char *str){
    struct tm tm;
    strptime(str,"%Y-%m-%d %H:%M:%S", &tm);  
    return mktime(&tm);
}

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

typedef struct segment {
    float x;
    float y;
    float r;
    int t;
    float nx;
    float ny;
    float nr;
    int nt;
} segment;


float presence[20][20][48];
int xmin = 600000;
int xmax = 604000;
int ymin = 2424000;
int ymax = 2428000;
int step = 200;
int xcount = 20;
int ycount = 20;
int tcount = 48;

static float norm(segment *seg){
    float x2 = (seg->x - seg->nx)*(seg->x - seg->nx);
    float y2 = (seg->y - seg->ny)*(seg->y - seg->ny);
    return sqrt(x2+y2);
}


static float uniform_random(float a,float b) {
    double r = random();
    return a+ (b-a) * r / ((double)RAND_MAX + 1);
}

static void circle_random(float x,float y,float r,float *ptrx,float *ptry){
    float angle = uniform_random(0,2*3.1415);
    float radius  = uniform_random(0.,r);
    *ptrx = x + radius*cos(angle);
    *ptry = y + radius*sin(angle);
    return;
}



static void test_uniform(){
    int i;
    for (i=0;i<100;i++)
        printf("uniform value [15,30] = %f\n",uniform_random(15,30));
    return; 
}

static void parse_line(char *line,unsigned int len,segment *s,float *scaling)
{
    char *param=line,*ptr=line;
    char *eol = line + len;
    int column_id = 0;
    // scan line until end of line
    while (ptr <= eol){
        // check for a parameter delimiter
        if ((*ptr == '\t') || (*ptr == '\n') || (*ptr == '\0'))
        {
            switch (column_id)
            {
                case 0:
                    // aimsi
                    break;
                case 1:
                    // dat_heur_debt
                    s->t = parse_date(param);
                    break;
                case 2:
                    // ndat_heur_debt
                    s->nt = parse_date(param);
                    break;
                case 3:
                    // mcc
                    break;
                case 4:
                    // calc_rayon
                    s->r = atof(param);
                    break;
                case 5:
                    // smooth_x
                    s->x = atof(param);
                    break;
                case 6:
                    // smooth_y
                    s->y = atof(param);
                    break;
                case 7:
                    // numr_cell
                    break;
                case 8:
                    // ncalc_rayon
                    s->nr = atoi(param);
                    break;
                case 9:
                    // nsmooth_x
                    s->nx = atof(param);
                    break;
                case 10:
                    // nsmooth_y
                    s->ny = atof(param);
                    break;
                case 11:
                    // scaling
                    *scaling = atof(param);
                    break;
            }
            column_id++;
            param = ptr+1;
        }
        ptr++;
    }
}

static void init_presence(void){
    int i,j,t;
    for (i=0;i<xcount;i++)
        for (j=0;j<ycount;j++)
            for (t=0;t<tcount;t++)
                presence[i][j][t] = 0.f;
    return;
}

static int intersect_points(segment *seg,segment *iseg){
    // return intersect_points as a segment iseg
    // -1 if no intersection
    // 
    float d = norm(seg);

    if (d > (seg->r + seg->nr))
        return -1;
    else if ((d < fabs(seg->r - seg->nr)) || (d == 0))
        return 0;

    float a = (seg->r*seg->r - seg->nr*seg->nr + d*d) / (2 * d);
    float h = sqrt(seg->r*seg->r - a*a);
    float p2x = seg->x + a*(seg->nx - seg->x)/d;
    float p2y = seg->y + a*(seg->ny - seg->y)/d;

    iseg->x = p2x + h * (seg->ny - seg->y) / d;
    iseg->y = p2y - h * (seg->nx - seg->x) / d;

    iseg->nx = p2x - h * (seg->ny - seg->y) / d;
    iseg->ny = p2y + h * (seg->nx - seg->x) / d;
    return 1;
}


static void randsample_moving_position(segment *seg,float p,float *ptrx,float *ptry){
    float xa,ya,xb,yb;
    circle_random(seg->x,seg->y,seg->r,&xa,&ya);
    circle_random(seg->nx,seg->ny,seg->nr,&xb,&yb);
    *ptrx = xa + (xa-xb)*p;
    *ptry = ya + (ya-yb)*p;
    return;
}

static int randsample_static_position(segment *seg,float *ptrx,float *ptry){
    // return 0 if no static position can be found, >0 instead with position *ptrx,*ptry
    segment iseg;
    float box[4];
    float x,y;
    segment sega,segb;

    int ret = intersect_points(seg,&iseg);

    if (ret < 0)
        return 0;
    if (ret == 0){
        // one circle is fully inside the other, let's take the smallest one
        if (seg->r <seg->nr){
            box[0] = seg->x - seg->r;
            box[1] = seg->x + seg->r;
            box[2] = seg->y - seg->r;
            box[3] = seg->y + seg->r;
        }else{
            box[0] = seg->nx - seg->nr;
            box[1] = seg->nx + seg->nr;
            box[2] = seg->ny - seg->nr;
            box[3] = seg->ny + seg->nr;
        }
    }else{
        // circle to circle intersection
        float hx = iseg.x+iseg.nx;
        float hy = iseg.y+iseg.ny;
        float d = norm(&iseg)/2.;
        box[0] = hx-d;
        box[1] = hx+d;
        box[2] = hy-d;
        box[3] = hy+d;
    }
    int retry = 0;
    sega.x = seg->x;
    sega.y = seg->y;
    segb.x = seg->x;
    segb.y = seg->y;

    while (retry++<25){
        x = uniform_random(box[0],box[1]);
        y = uniform_random(box[2],box[3]);
        sega.nx = x;
        sega.ny = y;
        segb.nx = x;
        segb.ny = y;
        float da = norm(&sega);
        float db = norm(&segb);
        if ((da<seg->r) && (db<seg->nr)){
            // x,y is in the intersection of 
            // circle(seg->x,seg->y,seg->r) and circle(seg->nx,seg->ny,seg->nr)
            *ptrx = x;
            *ptry = y;
            return 1;
        }
    }
    *ptrx = (box[0]+box[1])/2.;
    *ptry = (box[2]+box[3])/2.;
    return 2;
}

static void test_intersection(void){
    segment is,os;

    is.x = 0;
    is.y = 0;
    is.nx = 1;
    is.ny = 1;
    is.r = 2;
    is.nr = 2;
    intersect_points(&is,&os);
    printf("Intersection:(%f,%f) (%f,%f)\n",os.x,os.y,os.nx,os.ny); 
 
    is.x = 0;
    is.y = 0;
    is.nx = 4;
    is.ny = 0;
    is.r = 2;
    is.nr = 2;
    intersect_points(&is,&os);
    printf("Single-point edge collision:(%f,%f) (%f,%f)\n",os.x,os.y,os.nx,os.ny);

    is.x = 0;
    is.y = 0;
    is.nx = 1;
    is.ny = 0;
    is.r = 5;
    is.nr = 2;
    printf("Wholly inside: %d\n", intersect_points(&is,&os));

    is.x = 0;
    is.y = 0;
    is.nx = 5;
    is.ny = 0;
    is.r = 2;
    is.nr = 2;
    printf("No collision: %d\n", intersect_points(&is,&os));
    return;
}

static void interpolate(segment *seg,int now){
    if ((now < seg->t) || (now >=seg->nt))
        return;
    // compute percentage
    float p = (now - seg->t)*1.0f/(seg->nt-seg->t);

    // for a circle_to_cirle intersection 
    // leading to a static position insight
    // allow :
    // - a static drift speed of 1m.s-1
    // - a random walking with maximum displacement ~ sqrt(time)
    // - never exceeding twice the original cell radius
    float ra = MIN(seg->r + sqrt(now-seg->t)*1., 2. * seg->r);
    float rb = MIN(seg->nr + sqrt(now)*1., 2 * 2. * seg->nr);
    float x,y;
    int ret = randsample_static_position(seg,&x,&y);
    if (ret == 0){
        randsample_moving_position(seg,p,&x,&y);
    }
    return;
}



static void parse_file(char *filename){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1) {
        segment seg;
        float scaling;
        parse_line(line,len,&seg,&scaling);

        int rep,hour;
        char date[64];
        for (hour=4;hour<23;hour++){
            sprintf(date,"2014-05-19 %02d:00:00",hour);
            int now = parse_date(date);
            for(rep=0;rep<25;rep++)
            {
                interpolate(&seg,now);
            }
        }
        //printf("==> %f,%f,%f,%d,%f,%f,%f,%d,%f\n",seg.x,seg.y,seg.r,seg.t,seg.nx,seg.ny,seg.nr,seg.nt,scaling);
    }

    fclose(fp);
    if (line)
        free(line);
    exit(EXIT_SUCCESS);

}

int main(int argc,char *argv[])
{
    init_presence();
//    test_intersection();
//    test_uniform();
    parse_file("/home/ngaude/workspace/data/arzephir_italy_place_segment_2014-05-19.tsv");
    return 0;
}



