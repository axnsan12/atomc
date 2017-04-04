struct Pt{
	int x,y[];
	double z;
	};

struct Pt points[20/4+5];

int		count()
{
	int		i,n,redef;
	for(i=n=0;i<10;i=i+1){
	    double redef;
		if(points[i].x>=0&&points[i].y>=0) {
	        char points;
	        struct Pt redef;
            n=n+1;
		}
	}
	return n;
}

void main()
{
	put_i(count());
}
