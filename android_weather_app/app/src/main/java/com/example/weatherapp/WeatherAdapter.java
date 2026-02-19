package com.example.weatherapp;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.util.List;

public class WeatherAdapter extends RecyclerView.Adapter<WeatherAdapter.ViewHolder> {

    private List<String> times;
    private List<Double> tempsMV;
    private List<Double> tempsPA;

    public WeatherAdapter(List<String> times, List<Double> tempsMV, List<Double> tempsPA) {
        this.times = times;
        this.tempsMV = tempsMV;
        this.tempsPA = tempsPA;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_weather, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        String time = times.get(position);
        if (time.contains("T")) {
            time = time.split("T")[1];
        }

        holder.tvTime.setText(time);

        if (position < tempsMV.size()) {
            holder.tvMvTemp.setText(String.format("%.1f°C", tempsMV.get(position)));
        } else {
            holder.tvMvTemp.setText("");
        }

        if (position < tempsPA.size()) {
            holder.tvPaTemp.setText(String.format("%.1f°C", tempsPA.get(position)));
        } else {
            holder.tvPaTemp.setText("");
        }
    }

    @Override
    public int getItemCount() {
        return times != null ? times.size() : 0;
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView tvTime;
        TextView tvMvTemp;
        TextView tvPaTemp;

        ViewHolder(View itemView) {
            super(itemView);
            tvTime = itemView.findViewById(R.id.tv_time);
            tvMvTemp = itemView.findViewById(R.id.tv_mv_temp);
            tvPaTemp = itemView.findViewById(R.id.tv_pa_temp);
        }
    }
}
