package com.example.weatherapp;

public class WeatherResponse {
    private double latitude;
    private double longitude;
    private Hourly hourly;

    public double getLatitude() { return latitude; }
    public double getLongitude() { return longitude; }
    public Hourly getHourly() { return hourly; }
}
