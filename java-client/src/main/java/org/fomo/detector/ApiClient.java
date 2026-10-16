package org.fomo.detector;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;

final class ApiClient {
    private final String baseUrl;

    ApiClient(String baseUrl) {
        String normalized = baseUrl == null ? "" : baseUrl.trim();
        while (normalized.endsWith("/")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        if (!normalized.startsWith("http://") && !normalized.startsWith("https://")) {
            throw new IllegalArgumentException("The service address must begin with http:// or https://.");
        }
        this.baseUrl = normalized;
    }

    JsonObject health() throws IOException {
        return request("GET", "/api/v1/health", null);
    }

    JsonObject train() throws IOException {
        return request("POST", "/api/v1/train", "{}" );
    }

    JsonObject startSimulation(double durationSeconds) throws IOException {
        JsonObject body = new JsonObject();
        body.addProperty("duration_seconds", durationSeconds);
        return request("POST", "/api/v1/simulations", body.toString());
    }

    JsonObject simulation(String sessionId) throws IOException {
        return request("GET", "/api/v1/simulations/" + sessionId, null);
    }

    JsonObject stopSimulation(String sessionId) throws IOException {
        return request("POST", "/api/v1/simulations/" + sessionId + "/stop", "{}" );
    }

    private JsonObject request(String method, String path, String body) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) URI.create(baseUrl + path).toURL().openConnection();
        connection.setRequestMethod(method);
        connection.setConnectTimeout(4000);
        connection.setReadTimeout(15000);
        connection.setRequestProperty("Accept", "application/json");

        if (body != null) {
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            byte[] requestBytes = body.getBytes(StandardCharsets.UTF_8);
            connection.setFixedLengthStreamingMode(requestBytes.length);
            try (var output = connection.getOutputStream()) {
                output.write(requestBytes);
            }
        }

        int statusCode = connection.getResponseCode();
        var responseStream = statusCode >= 400 ? connection.getErrorStream() : connection.getInputStream();
        String response;
        if (responseStream == null) {
            response = "{}";
        } else {
            try (responseStream) {
                response = new String(responseStream.readAllBytes(), StandardCharsets.UTF_8);
            }
        }
        connection.disconnect();

        JsonObject json;
        try {
            json = JsonParser.parseString(response).getAsJsonObject();
        } catch (RuntimeException exception) {
            throw new IOException("The service returned an invalid JSON response.", exception);
        }
        if (statusCode >= 400) {
            String message = json.has("detail") ? json.get("detail").getAsString() : "HTTP " + statusCode;
            throw new IOException(message);
        }
        return json;
    }
}