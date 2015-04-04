#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

/*********************************/

#if 1
#define XMIN        600000
#define XMAX        604000
#define YMIN        2424000
#define YMAX        2428000
#define XYSTEP      100
#define TSTEP       1800
#define REPEAT      100
#define WALK_SPEED  1.f
#define CNIL_THRESHOLD  100
#else
#define XMIN        0
#define XMAX        2000
#define YMIN        0
#define YMAX        2000
#define XYSTEP      50
#define TSTEP       3600
#define REPEAT      100000
#define WALK_SPEED  1.f
#define CNIL_THRESHOLD  0

#endif

/*********************************/

#define XCOUNT (((XMAX)-(XMIN))/(XYSTEP))
#define YCOUNT (((YMAX)-(YMIN))/(XYSTEP))
#define TCOUNT ((24*3600)/(TSTEP))

#define IDX(X) ((int)(((X)-XMIN)/XYSTEP))
#define IDY(Y) ((int)(((Y)-YMIN)/XYSTEP))
#define IDT(T) ((int)(((T)-day_start)/TSTEP))

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

/*********************************/

typedef struct segment {
    float x;
    float y;
    float r;
    time_t t;
    float nx;
    float ny;
    float nr;
    time_t nt;
} segment;

time_t day_start;
float presence[XCOUNT][YCOUNT][TCOUNT];

/*********************************/

static int parse_date(char *str){
    struct tm tm;
    strptime(str,"%Y-%m-%d %H:%M:%S", &tm);
    //printf("[%s=%d]\n",str,(int)timegm(&tm));
    return timegm(&tm);
}

static const char * timetodate(time_t t){
    static char date[64];
    struct tm *tm = gmtime(&t);
    strftime(date, 64,"%Y-%m-%d %H:%M:%S",tm);
    //printf("[%s=%d]\n",date,(int)t);
    return (const char *)&date;
}

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
#if 0
    float angle = uniform_random(0,2*3.1415);
    float radius  = uniform_random(0.,r);
    *ptrx = x + radius*cos(angle);
    *ptry = y + radius*sin(angle);
    return;
#else
    int retry = 0;
    segment seg;
    seg.x = x;
    seg.y = y;

    while (retry++<25){
        seg.nx = uniform_random(x-r,x+r);
        seg.ny = uniform_random(y-r,y+r);
        float d = norm(&seg);
        *ptrx = seg.nx;
        *ptry = seg.ny;
        if (d < r){
            // x,y is in the intersection of 
            // circle(seg->x,seg->y,seg->r) and circle(seg->nx,seg->ny,seg->nr)
            return;
        }
    }
    return;
#endif
}

static void parse_line(char *line,unsigned int len,segment *s,float *scaling)
{
    char *param=line,*ptr=line;
    char *eol = &line[len];
    int column_id = 0;
    // scan line until end of line
    while ((ptr <= eol)){
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
                    
                    /* last parameter */
                    return;
                    break;
            }
            if ((*ptr == '\n') || (*ptr == '\0'))

            {
                // end of line
                return;
            }
            column_id++;
            param = ptr+1;
        }
        ptr++;
    }
    exit(0);
}

static void zero_presence(void){
    int i,j,t;
    for (i=0;i<XCOUNT;i++)
        for (j=0;j<YCOUNT;j++)
            for (t=0;t<TCOUNT;t++)
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
    *ptrx = xa + (xb-xa)*p;
    *ptry = ya + (yb-ya)*p;
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
        if (seg->r < seg->nr){
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
        float hx = (iseg.x+iseg.nx)/2;
        float hy = (iseg.y+iseg.ny)/2;
        float d = norm(&iseg)/2.;
        box[0] = hx-d;
        box[1] = hx+d;
        box[2] = hy-d;
        box[3] = hy+d;
    }
    int retry = 0;
    sega.x = seg->x;
    sega.y = seg->y;
    segb.x = seg->nx;
    segb.y = seg->ny;

    while (retry++<25){
        x = uniform_random(box[0],box[1]);
        y = uniform_random(box[2],box[3]);
        sega.nx = x;
        sega.ny = y;
        segb.nx = x;
        segb.ny = y;
        float da = norm(&sega);
        float db = norm(&segb);
        *ptrx = x;
        *ptry = y;
        if ((da < seg->r) && (db < seg->nr)){
            // x,y is in the intersection of 
            // circle(seg->x,seg->y,seg->r) and circle(seg->nx,seg->ny,seg->nr)
            return 1;
        }
    }
//    *ptrx = (box[0]+box[1])/2.;
//    *ptry = (box[2]+box[3])/2.;
    return 2;
}


static void interpolate(segment *seg,int now,float *x,float *y){
    segment static_seg = {0,0,0,0,0,0,0,0};
    // compute percentage
    float p = (now - seg->t)*1.0f/(seg->nt-seg->t);

    // for a circle_to_cirle intersection 
    // leading to a static position insight
    // defines as :
    // - as a random walk
    // - with distance proportionnal ~ sqrt(time)
    // - with speed constant WALK_SPEED m.s-1
    static_seg = *seg;
    static_seg.r = seg->r + sqrt(now-seg->t)*WALK_SPEED;
    static_seg.nr = seg->nr + sqrt(seg->nt-now)*WALK_SPEED;
    int ret = randsample_static_position(&static_seg,x,y);
    if (ret == 0){
        randsample_moving_position(seg,p,x,y);
    }
    return;
}

static void parse_file(char *filename){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    char buff[64];

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    char *c = filename;
    while (*c++!='.' ){}
    memcpy(buff,c-11,10);
    memcpy(buff+10," 04:30:00",9);
    buff[19]='\0';
    day_start = parse_date(buff) + TSTEP; 
    time_t day_end = parse_date(buff) + 3600*24; 
    printf("begin at %s\n",timetodate(day_start));

    read = getline(&line, &len, fp);

    while ((read = getline(&line, &len, fp)) != -1) 
    {
        segment seg;
        float scaling = 1.;
        parse_line(line,len,&seg,&scaling);
        time_t t_start,t;
        t_start = day_start + (seg.t - day_start + TSTEP - 1)/TSTEP*TSTEP;

//        printf("| %f,%f,%f,%f,%f,%f,%f\n",seg.x,seg.y,seg.r,seg.nx,seg.ny,seg.nr,scaling);
//        printf("| %d,%s\n",(int)seg.t,timetodate(seg.t));
//        printf("| %d,%s\n",(int)seg.nt,timetodate(seg.nt));
//        printf("| %d,%s\n",(int)t_start,timetodate(t_start));

        int rep;
        int ix,iy,it;

        for (t = t_start;(t < seg.nt)||(t==day_end);t+=TSTEP)
        {
            it = IDT(t);
//            printf("%d<%d<%d\n",(int)t_start,(int)t,(int)seg.nt);

            if ((it>=TCOUNT) || (it<0))
                continue;
            for(rep=0;rep<REPEAT;rep++)
            {
                float x=0,y=0;
                interpolate(&seg,t,&x,&y);
                ix = IDX(x);
                iy = IDY(y);
                const char * date = 0;
                date = timetodate(t);
                //printf("#%d === %f|%f|%s %d,%d,%d\n",rep,x,y,date,ix,iy,it);
                char inside = (x<XMAX) & (y<YMAX) & (x>XMIN) & (y>YMIN);
                inside &= (ix<XCOUNT) & (ix>=0) & (iy<YCOUNT) & (iy>=0);

                if (inside)
                {
                    presence[ix][iy][it] += 1.*scaling/(1.*REPEAT);
                }
            }
        }
    }

    fclose(fp);
    if (line)
        free(line);
    return;
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

static void test_uniform(){
    int i;
    for (i=0;i<100;i++)
        printf("uniform value [15,30] = %f\n",uniform_random(15,30));
    return; 
}

static void test_date(){
    char *date = "2014-12-14 03:00:00";
    const char *pdate = NULL;
    time_t t = parse_date(date);
    pdate = timetodate(t);
    printf("%s == %s\n",date,pdate);
}

static void dump_presence(char *filename){
    FILE * fp;
    const char * date = 0;

    fp = fopen(filename, "a+");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    int ix,iy,it,x,y;
    int value;
    time_t t;
    for (it=0;it<TCOUNT;it++)
    {
        t = it*TSTEP + day_start;
        date = timetodate(t);
        for (iy=0;iy<YCOUNT;iy++)
        {
            y = iy*XYSTEP+YMIN;
            for (ix=0;ix<XCOUNT;ix++)
            {
                x = ix*XYSTEP+XMIN;
                value = presence[ix][iy][it];
                if (value < CNIL_THRESHOLD)
                    value = 0;
                fprintf(fp,"%s,%d,%d,%d,%d,%d\n",date,x,y,x+XYSTEP,y+XYSTEP,value);
            }
        }
    }
    fclose(fp);
    return;
}

void test_presence(){
    float maxv;
    float count;
    float sumv;
    int ix,iy,it;
    time_t t;
    const char *date;
    for (it=0;it<TCOUNT;it++)
    {
        count = 0;
        sumv = 0;
        maxv = -1000000000;
        for (ix=0;ix<XCOUNT;ix++)
        {
            for (iy=0;iy<YCOUNT;iy++)
            {
                float p = presence[ix][iy][it];
                if (p > 0)
                {
                    count++;
                    maxv = MAX(maxv,p);
                    sumv += presence[ix][iy][it];
                }
            }
        }
        t = it*TSTEP + day_start;
        date = timetodate(t);
        printf("presence %s max=%f, avg=%f, sum=%f\n",date,maxv,sumv/count,sumv);
    }
}

int main(int argc,char *argv[])
{
    char *in_filename = 0;
    char *out_filename = 0;
    //srandom(151175);
    srandom(151);
    //test_date();
    //test_intersection();
    //test_uniform();
    if (argc >1)
        in_filename = argv[1];
    else 
        in_filename = "presence_test_1975-11-15.tsv";

    if (argc >2)
        out_filename = argv[2];
    else 
        out_filename = "presence.csv";

  
    zero_presence();
    parse_file(in_filename);
    dump_presence(out_filename);
    test_presence();
    return 0;
}



