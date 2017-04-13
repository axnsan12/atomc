struct Pt{
	int x,y;
	double z;
	};

struct Pt points[20/4+5];

int		count()
{
	int		i,n,redef;
	for(i=n=0;i<10;i=i+1){
	    points[i].x = points[i].y = i + 1;
	    points[i].z = 1.414213 * points[i].x;
	}
	for(i=n=0;i<10;i=i+1){
	    double redef;
		if(points[i].x>=0&&points[i].y>=0) {
            {
                char points;
                points = (char) i + '0';
                put_c(points);
                put_s(" ");
                n=(n+1)*2;
                put_i(n);
	        }
            struct Pt redef;
	        redef = points[i];
	        put_s(" ");
	        put_d(redef.z);
	        put_s("\n");
		}
	}
	return n;
}

void main()
{
	put_i(count());
}
