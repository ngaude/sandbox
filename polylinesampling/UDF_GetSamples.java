package polylinesampling;

import com.esri.core.geometry.SpatialReference;
import com.esri.core.geometry.ogc.OGCGeometry;
import com.esri.core.geometry.Geometry;
import com.esri.core.geometry.Polyline;
import com.esri.core.geometry.SegmentIterator;
import com.esri.core.geometry.Segment;

import java.util.ArrayList;

import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.udf.UDFType;

import java.util.Date;
import java.text.SimpleDateFormat;
import java.text.ParseException;


@UDFType(stateful = false)
@Description(
		name = "GetSamples", 
		value = "_FUNC_(String lineString,int sampleCount) \n" +
				"- String[] lineString : String array for concat of TIMESTAMP+NUMR_CELL+X+Y+RADIUS \n"+
				"- int delta : Maximun difference between two events", 
		extended = ""
		)

public class GetSamples extends UDF {

	
	public static final String rennes_stmalo_wkt = "LINESTRING (279902.072878999984823 2414224.99127299990505,281469.065775999973994 2413613.202874000184238,282724.000607999972999 2412109.585442999843508,286076.284336999990046 2410390.721078000031412,288628.264171999995597 2409969.885460999794304,294075.668120999995153 2406019.733884999994189,298229.413243999995757 2403010.151095000095665,298506.00580899999477 2402085.134173000231385,297945.518448000017088 2400677.262263999786228,296923.782066999992821 2399774.031332999933511,295681.005948000005446 2396598.806811000220478,296965.097690999973565 2393953.285813999827951,297232.738692999992054 2392069.467703999951482,298188.827829000016209 2390904.601768999826163,298085.821776999975555 2388922.541220000013709,300111.151286999986041 2386383.199959000106901,300651.800004000018816 2384828.224636999890208,301177.088538000010885 2384354.665473000146449,301761.998160000017378 2382659.984029000159353,301484.372342000016943 2380821.240327999927104,301985.199352999974508 2379861.237340999767184,302086.963153999997303 2377072.483781000133604,302635.441435999993701 2374640.810806999914348,302378.704870000015944 2372808.609958000015467,303038.420564999978524 2370944.318028999958187,302864.250440999981947 2369528.981145999860018,302968.865369000006467 2367976.97037100000307,304480.138590999995358 2366780.629399000201374,304661.41681000002427 2364715.003291999921203,305511.697614000004251 2363755.557742999866605,305631.93967499997234 2362089.316963000223041,304581.601175000017975 2360716.086837000213563,303336.127987999992911 2357925.580486000049859,301429.099318999971729 2357023.633318999782205,300452.704120000009425 2355931.546843999996781,300242.622501000005286 2354228.792291000019759,299326.275775999994949 2353520.932713000103831,299637.226594000007026 2352402.981565000023693,301580.926705999998376 2352385.0723069999367)";
	public static final String test1 = "LINESTRING ( 0 0 , 11 11 , 33 33 , 100 100)";
	public static final String test2 = "LINESTRING ( 0 0 , 50 50 , 100 100)";
	
	public static void main(String[] args) throws Exception {
		// display trajectory for train 8083 on 1st of april 2014 : 12:24,Rennes,13:05,Saint-Malo 
		// with a +/-5 minutes margin
		displaySampledTrain("8085","2014-07-01 12:22:00","2014-07-01 13:10:00",5,rennes_stmalo_wkt);
	}
	
	public static void displaySampledTrain(String trainId, String begin, String end,int margin, String wkt) throws ParseException 
	{
		SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		Date beginDate = sdf.parse(begin);
		Date endDate = sdf.parse(end);
		
		
		
		int sampleCount = (int)(endDate.getTime() -  beginDate.getTime())/(1000 * 60) + 1;
		ArrayList<String> sampleList = sampleLineString(wkt,sampleCount);
		
		long currTime = beginDate.getTime();
		System.out.println("trainId,dat_heur,x,y");
		for (String sample: sampleList){
			for (int i=-margin ; i < margin ; i++){
				String currDateString =sdf.format(new Date(currTime+60000*i)); 
				System.out.println(trainId + "," + currDateString + "," + sample);
			}
			currTime += 60000;
		}
		
		return;
	}
	
	public ArrayList<String> evaluate(final Text text, int sampleCount) {
		
		ArrayList<String> sampleList;
		String wkt = text.toString();
		
		sampleList = sampleLineString(wkt,sampleCount);
		
	    
	    return sampleList;
	}
	
	  public int evaluate(double xa, double ya,double xb,double yb,double x,double y)
	  {
	    if ( (x > Math.min(xa,xb)) && ( x < Math.max(xa,xb) ) && ( y > Math.min(ya,yb) ) && ( x < Math.max(ya,yb) ) ) return 1;
	    return 0 ;
	  }

	
	public static ArrayList<String> sampleLineString(String wkt, int sampleCount) {

		ArrayList<String> sampleList = new ArrayList<String>();
		
		if (sampleCount < 2)
			return sampleList;

		OGCGeometry ogcObj = null;
		Geometry geom = null;
		Polyline pl = null;
		try {
			SpatialReference spatialReference = null;
			ogcObj = OGCGeometry.fromText(wkt);
			ogcObj.setSpatialReference(spatialReference);
			geom = ogcObj.getEsriGeometry();
			if (geom.getType() != Geometry.Type.Polyline) {
				/* shall be a polyline e.g a linestring() */
				return sampleList;
			}
			pl = (Polyline) geom;
		} catch (Exception e) {
			System.out.println(e);
		}
		
		SegmentIterator it = pl.querySegmentIterator();
		it.nextPath();
		double totalLength = pl.calculateLength2D();
		int sampleId = 0;
		double samplePos = 0;
		double sampleStep = totalLength/(sampleCount-1);
		Segment seg = null;
		while (it.hasNextSegment()) {
			seg = it.nextSegment();
			double segLength = seg.calculateLength2D();
			
			if (segLength > 0) {
				
				while ((samplePos <= segLength) && (sampleId < sampleCount)) {
					double xa = seg.getStartX();
					double ya = seg.getStartY();
					double xb = seg.getEndX();
					double yb = seg.getEndY();
				
					double xs = Math.round(xa + (xb-xa)*samplePos/segLength);
					double ys = Math.round(ya + (yb-ya)*samplePos/segLength);
					
					// add a new sample to the list
					String sample = xs + "," + ys; 
					sampleList.add(sample);
					sampleId++;
					samplePos+= sampleStep;
				}
				samplePos -= segLength; 
			}
		}
		// to cope with rounding errors, force last sample at the end of the line string.
		while (sampleId < sampleCount){
			String sample = Math.round(seg.getEndX()) + "," + Math.round(seg.getEndY());
			sampleList.add(sample);
			sampleId++;
		}
		return sampleList;

	}

}
