package com.example.weatherapp;

import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class MainActivity extends AppCompatActivity {

    private RecyclerView recyclerView;
    private WeatherAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        recyclerView = findViewById(R.id.recycler_view);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        fetchWeather();
    }

    private void fetchWeather() {
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://api.open-meteo.com/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        WeatherService service = retrofit.create(WeatherService.class);

        Call<List<WeatherResponse>> call = service.getForecast(
                "37.3861,37.4419",
                "-122.0839,-122.1430",
                "temperature_2m",
                "America/Los_Angeles"
        );

        call.enqueue(new Callback<List<WeatherResponse>>() {
            @Override
            public void onResponse(Call<List<WeatherResponse>> call, Response<List<WeatherResponse>> response) {
                if (response.isSuccessful() && response.body() != null && response.body().size() >= 2) {
                    WeatherResponse mvData = response.body().get(0);
                    WeatherResponse paData = response.body().get(1);

                    List<String> times = mvData.getHourly().getTime();
                    List<Double> tempsMV = mvData.getHourly().getTemperature2m();
                    List<Double> tempsPA = paData.getHourly().getTemperature2m();

                    adapter = new WeatherAdapter(times, tempsMV, tempsPA);
                    recyclerView.setAdapter(adapter);
                } else {
                    Log.e("WeatherApp", "Response unsuccessful or empty");
                }
            }

            @Override
            public void onFailure(Call<List<WeatherResponse>> call, Throwable t) {
                Log.e("WeatherApp", "API Call failed", t);
            }
        });
    }
}
