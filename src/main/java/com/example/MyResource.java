package com.example;


import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

import javax.servlet.ServletContext;
import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.QueryParam;
import javax.ws.rs.core.Context;
import javax.ws.rs.core.MediaType;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayesModel;

import com.example.JsonModel;

/**
 * Root resource (exposed at "prediction" path)
 */

@Path("/prediction")
public class MyResource {
	
	@QueryParam("uniqueCarrier") String uniqueCarrier;
	@QueryParam("currentOrigin") String currentOrigin;
	@QueryParam("currentDest") String currentDest;
	@QueryParam("dateToPredict") String dateToPredict;
	@QueryParam("originTime") String originTime;
	@QueryParam("destTime") String destTime;
	
	@Context ServletContext context;
	
    /**
     * Method handling HTTP GET requests. The returned object will be sent
     * to the client as "APPLICATION_JSON" media type.
     *
     * @return String that will be returned as a APPLICATION_JSON response.
     * @throws ParseException 
     */
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public JsonModel getIt() throws ParseException {
    	SparkConf conf = new SparkConf().setAppName("Predictions").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        //System.out.println(sc.getLocalProperty("akka.version").toString());
        NaiveBayesModel model01 = NaiveBayesModel.load(sc.sc(), "naiveBayesModel01");
        NaiveBayesModel modelCategory = NaiveBayesModel.load(sc.sc(), "naiveBayesModelCategory");
    	
    	Flight flight = new Flight();
    	
    	String[] partsDate = dateToPredict.split("\\-");
        flight.setYear(Integer.parseInt(partsDate[0]));
        flight.setMonth(Integer.parseInt(partsDate[1]));
        flight.setDayofMonth(Integer.parseInt(partsDate[2]));
        
        Calendar c = Calendar.getInstance();
        DateFormat df = new SimpleDateFormat("yyyy-mm-dd"); //this is the format that I receive from Angular
        Date date = df.parse(dateToPredict);
        c.setTime(date);
        flight.setDayOfWeek(c.get(Calendar.DAY_OF_WEEK));
        
        flight.setUniqueCarrier(uniqueCarrier);
        flight.setOrigin(currentOrigin);
        flight.setDest(currentDest);
        
        String[] partsTime;
        String time = "";
        partsTime = originTime.split(":");
        time = partsTime[0].concat(partsTime[1]);
        flight.setCRSDepTime(Integer.parseInt(time));
        
        partsTime = destTime.split(":");
        time = partsTime[0].concat(partsTime[1]);
        flight.setCRSArrTime(Integer.parseInt(time));
    	
    	//MAKE A PREDICTION
        double prediction = model01.predict(flight.getVectorFeaturesV2());
        double[] probabilities = modelCategory.predictProbabilities(flight.getVectorFeaturesV2()).toArray();
        
        
    	//Make the JSON Result
    	JsonModel result = new JsonModel();
    	result.setUniqueCarrier(uniqueCarrier);
    	result.setOrigin(currentOrigin);
    	result.setDest(currentDest);
    	result.setDateToPredict(dateToPredict);
    	result.setOriginTime(originTime);
    	result.setDestTime(destTime);
    	result.setPrediction(prediction);
    	
	    if(prediction == 0.0){
	    	result.setProb0(probabilities[0]*100);
	    	result.setProb0_20(probabilities[1]*100);
	    	result.setProb20_90(probabilities[2]*100);
	    	result.setProb90plus(probabilities[3]*100);
	    } else {
	    	double sum = probabilities[1] + probabilities[2] + probabilities[3];
	    	result.setProb0(0.0);
	    	result.setProb0_20((probabilities[1]/sum)*100);
	    	result.setProb20_90((probabilities[2]/sum)*100);
	    	result.setProb90plus((probabilities[3]/sum)*100);
	    }
    	
    	sc.close();
    	
    	System.out.println("Nuova richiesta servita...");
    	return result;
    }
}
