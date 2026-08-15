package com.example.weatherapp;

import java.util.List;
import retrofit2.Call;
import retrofit2.http.GET;
import retrofit2.http.Query;

public interface WeatherService {
    @GET("v1/forecast")
    Call<List<WeatherResponse>> getForecast(
        @Query("latitude") String latitude,
        @Query("longitude") String longitude,
        @Query("hourly") String hourly,
        @Query("timezone") String timezone
    );
}
