package com.example.weatherapp;

import java.util.List;
import com.google.gson.annotations.SerializedName;

public class Hourly {
    private List<String> time;
    @SerializedName("temperature_2m")
    private List<Double> temperature2m;

    public List<String> getTime() { return time; }
    public List<Double> getTemperature2m() { return temperature2m; }
}
