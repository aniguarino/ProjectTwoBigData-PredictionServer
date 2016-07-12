package com.example;

import java.io.Serializable;

public class JsonModel implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private String uniqueCarrier;
	private String origin;
	private String dest;
	private String dateToPredict;
	private String originTime;
	private String destTime;
	private Double prediction;
	private Double prob0;
	private Double prob0_20;
	private Double prob20_90;
	private Double prob90plus;
	
	public void setDateToPredict(String dateToPredict) {
		String[] parts = dateToPredict.split("\\-");
		String date = "";
		
		if(parts[1].equals("01"))
    		date = parts[2]+" Gennaio "+parts[0];
    	if(parts[1].equals("02"))
    		date = parts[2]+" Febbraio "+parts[0];
    	if(parts[1].equals("03"))
    		date = parts[2]+" Marzo "+parts[0];
    	if(parts[1].equals("04"))
    		date = parts[2]+" Aprile "+parts[0];
    	if(parts[1].equals("05"))
    		date = parts[2]+" Maggio "+parts[0];
    	if(parts[1].equals("06"))
    		date = parts[2]+" Giugno "+parts[0];
    	if(parts[1].equals("07"))
    		date = parts[2]+" Luglio "+parts[0];
    	if(parts[1].equals("08"))
    		date = parts[2]+" Agosto "+parts[0];
    	if(parts[1].equals("09"))
    		date = parts[2]+" Settembre "+parts[0];
    	if(parts[1].equals("10"))
    		date = parts[2]+" Ottobre "+parts[0];
    	if(parts[1].equals("11"))
    		date = parts[2]+" Novembre "+parts[0];
    	if(parts[1].equals("12"))
    		date = parts[2]+" Dicembre "+parts[0];
		
		this.dateToPredict = date;
	}

	public String getUniqueCarrier() {
		return uniqueCarrier;
	}

	public void setUniqueCarrier(String uniqueCarrier) {
		this.uniqueCarrier = uniqueCarrier;
	}

	public String getOrigin() {
		return origin;
	}

	public void setOrigin(String origin) {
		this.origin = origin;
	}

	public String getDest() {
		return dest;
	}

	public void setDest(String dest) {
		this.dest = dest;
	}

	public String getOriginTime() {
		return originTime;
	}

	public void setOriginTime(String originTime) {
		this.originTime = originTime;
	}

	public String getDestTime() {
		return destTime;
	}

	public void setDestTime(String destTime) {
		this.destTime = destTime;
	}

	public Double getPrediction() {
		return prediction;
	}

	public void setPrediction(Double prediction) {
		this.prediction = prediction;
	}

	public Double getProb0() {
		return prob0;
	}

	public void setProb0(Double prob0) {
		this.prob0 = prob0;
	}

	public Double getProb0_20() {
		return prob0_20;
	}

	public void setProb0_20(Double prob0_20) {
		this.prob0_20 = prob0_20;
	}

	public Double getProb20_90() {
		return prob20_90;
	}

	public void setProb20_90(Double prob20_90) {
		this.prob20_90 = prob20_90;
	}

	public Double getProb90plus() {
		return prob90plus;
	}

	public void setProb90plus(Double prob90plus) {
		this.prob90plus = prob90plus;
	}

	public String getDateToPredict() {
		return dateToPredict;
	}

	@Override
	public String toString() {
		return "JsonModel [uniqueCarrier=" + uniqueCarrier + ", origin=" + origin + ", dest=" + dest
				+ ", dateToPredict=" + dateToPredict + ", originTime=" + originTime + ", destTime=" + destTime
				+ ", prediction=" + prediction + ", prob0=" + prob0 + ", prob0_20=" + prob0_20 + ", prob20_90="
				+ prob20_90 + ", prob90plus=" + prob90plus + "]";
	}
}
