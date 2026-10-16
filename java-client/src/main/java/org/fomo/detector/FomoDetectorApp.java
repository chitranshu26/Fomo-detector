package org.fomo.detector;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.beans.property.ReadOnlyStringWrapper;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Spinner;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class FomoDetectorApp extends Application {
    private final ExecutorService requests = Executors.newFixedThreadPool(4);
    private final ScheduledExecutorService polling = Executors.newSingleThreadScheduledExecutor();
    private final AtomicBoolean pollInFlight = new AtomicBoolean();
    private final ObservableList<SampleRow> recentSamples = FXCollections.observableArrayList();

    private final TextField serviceAddress = new TextField("http://127.0.0.1:8765");
    private final Label connectionStatus = new Label("Not connected");
    private final Label modelStatus = new Label("Model status unknown");
    private final Label feedback = new Label("Connect to the Python engine to begin.");
    private final Label predictionValue = new Label("Waiting for samples");
    private final Label confidenceValue = new Label("--");
    private final Label sampleCount = new Label("0 samples");
    private final Button connectButton = new Button("Connect");
    private final Button trainButton = new Button("Train model");
    private final Button startButton = new Button("Start simulation");
    private final Button stopButton = new Button("Stop");
    private final Spinner<Integer> duration = new Spinner<>(5, 3600, 30);
    private final ProgressIndicator activity = new ProgressIndicator();
    private final XYChart.Series<Number, Number> scoreSeries = new XYChart.Series<>();
    private final XYChart.Series<Number, Number> thresholdSeries = new XYChart.Series<>();
    private final TableView<SampleRow> sampleTable = new TableView<>(recentSamples);

    private volatile String sessionId;
    private String lastRecordedAt = "";
    private int sampleIndex;

    @Override
    public void start(Stage stage) {
        BorderPane root = new BorderPane();
        root.getStyleClass().add("app-root");
        root.setTop(createHeader());
        root.setCenter(createContent());
        root.setBottom(createFooter());

        Scene scene = new Scene(root, 1180, 900);
        scene.getStylesheets().add(getClass().getResource("/styles.css").toExternalForm());
        stage.setTitle("FOMO Detector | Signal Lab");
        stage.setMinWidth(360);
        stage.setMinHeight(640);
        stage.setScene(scene);
        stage.setOnCloseRequest(event -> shutdown());
        stage.show();

        connect();
    }

    private VBox createHeader() {
        Label eyebrow = new Label("SIGNAL LAB  /  SYNTHETIC EEG");
        eyebrow.getStyleClass().add("eyebrow");
        Label title = new Label("FOMO Detector");
        title.getStyleClass().add("page-title");
        Label subtitle = new Label("Train, simulate, and inspect each model prediction.");
        subtitle.getStyleClass().add("subtitle");

        HBox titleRow = new HBox(18, new VBox(7, eyebrow, title, subtitle), spacer(), connectionStatus);
        titleRow.setAlignment(Pos.CENTER_LEFT);
        VBox header = new VBox(18, titleRow, createConnectionBar());
        header.getStyleClass().add("header");
        return header;
    }

    private HBox createConnectionBar() {
        serviceAddress.setPromptText("Python API address");
        serviceAddress.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(serviceAddress, Priority.ALWAYS);
        connectButton.getStyleClass().add("secondary-button");
        connectButton.setOnAction(event -> connect());
        HBox row = new HBox(10, new Label("ENGINE ADDRESS"), serviceAddress, connectButton);
        row.setAlignment(Pos.CENTER_LEFT);
        row.getStyleClass().add("connection-bar");
        return row;
    }

    private ScrollPane createContent() {
        Label sectionTitle = new Label("Workspace");
        sectionTitle.getStyleClass().add("section-title");
        Label sectionCaption = new Label("The Java client controls the existing Python model service.");
        sectionCaption.getStyleClass().add("section-caption");

        FlowPane cards = new FlowPane(16, 16);
        cards.setPrefWrapLength(1080);
        cards.getChildren().addAll(
                createTrainingCard(),
                createSimulationCard(),
                createChartCard(),
                createSamplesCard()
        );

            FlowPane metrics = new FlowPane(12, 12);
            metrics.getChildren().addAll(
                metricCard("LATEST PREDICTION", predictionValue),
                metricCard("MODEL PROBABILITY", confidenceValue),
                metricCard("SAMPLES THIS RUN", sampleCount)
            );

            VBox content = new VBox(16, sectionTitle, sectionCaption, metrics, cards);
        content.setPadding(new Insets(24, 16, 30, 16));
        ScrollPane scrollPane = new ScrollPane(content);
        scrollPane.setFitToWidth(true);
        scrollPane.getStyleClass().add("content-scroll");
        return scrollPane;
    }

    private VBox createTrainingCard() {
        Label title = new Label("Model");
        title.getStyleClass().add("card-title");
        modelStatus.getStyleClass().add("status-line");
        Label description = new Label("Train the Random Forest from the configured realistic dataset. Existing model files are replaced only after training completes.");
        description.getStyleClass().add("body-copy");
        description.setWrapText(true);
        trainButton.getStyleClass().add("primary-button");
        trainButton.setMaxWidth(Double.MAX_VALUE);
        trainButton.setOnAction(event -> train());
        VBox card = card(title, modelStatus, description, trainButton);
        card.setPrefWidth(390);
        return card;
    }

    private VBox createSimulationCard() {
        Label title = new Label("Simulation controls");
        title.getStyleClass().add("card-title");
        Label durationLabel = new Label("DURATION  /  SECONDS");
        durationLabel.getStyleClass().add("field-label");
        duration.setEditable(true);
        duration.setPrefWidth(130);
        duration.getEditor().setText("30");
        duration.valueProperty().addListener((observable, oldValue, newValue) ->
                duration.getEditor().setText(newValue.toString()));

        startButton.getStyleClass().add("primary-button");
        startButton.setOnAction(event -> startSimulation());
        stopButton.getStyleClass().add("danger-button");
        stopButton.setDisable(true);
        stopButton.setOnAction(event -> stopSimulation());
        activity.setPrefSize(24, 24);
        activity.setVisible(false);
        HBox actions = new HBox(10, startButton, stopButton, activity);
        actions.setAlignment(Pos.CENTER_LEFT);

        VBox card = card(title, durationLabel, duration, actions, feedback);
        card.setPrefWidth(390);
        return card;
    }

    private VBox createChartCard() {
        Label title = new Label("Live FOMO score");
        title.getStyleClass().add("card-title");
        Label caption = new Label("Generated score by sample; dashed guide marks the demo alert threshold.");
        caption.getStyleClass().add("body-copy");
        caption.setWrapText(true);

        NumberAxis xAxis = new NumberAxis();
        xAxis.setLabel("Sample");
        NumberAxis yAxis = new NumberAxis(0, 2.2, 0.2);
        yAxis.setLabel("Score");
        LineChart<Number, Number> chart = new LineChart<>(xAxis, yAxis);
        chart.setAnimated(false);
        chart.setLegendVisible(false);
        chart.setCreateSymbols(false);
        chart.setMinHeight(270);
        chart.setPrefHeight(300);
        scoreSeries.setName("FOMO score");
        thresholdSeries.setName("Alert threshold");
        thresholdSeries.getData().addAll(
                new XYChart.Data<>(0, 0.75),
                new XYChart.Data<>(100, 0.75)
        );
        chart.getData().addAll(scoreSeries, thresholdSeries);

        VBox card = card(title, caption, chart);
        card.setPrefWidth(710);
        return card;
    }

    private VBox createSamplesCard() {
        Label title = new Label("Recent predictions");
        title.getStyleClass().add("card-title");
        sampleCount.getStyleClass().add("section-caption");

        TableColumn<SampleRow, String> number = column("#", SampleRow::index, 48);
        TableColumn<SampleRow, String> actual = column("Generated", SampleRow::actual, 110);
        TableColumn<SampleRow, String> predicted = column("Predicted", SampleRow::predicted, 110);
        TableColumn<SampleRow, String> score = column("Score", SampleRow::score, 80);
        TableColumn<SampleRow, String> confidence = column("Confidence", SampleRow::confidence, 105);
        sampleTable.getColumns().setAll(number, actual, predicted, score, confidence);
        sampleTable.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY_FLEX_LAST_COLUMN);
        sampleTable.setPlaceholder(new Label("Simulation samples will appear here."));
        sampleTable.setPrefHeight(270);

        VBox card = card(title, sampleCount, sampleTable);
        card.setPrefWidth(710);
        return card;
    }

    private VBox createFooter() {
        Label disclaimer = new Label("Synthetic simulation only. Not a medical device or real EEG interpretation.");
        disclaimer.getStyleClass().add("disclaimer");
        HBox footer = new HBox(disclaimer);
        footer.setAlignment(Pos.CENTER_LEFT);
        footer.getStyleClass().add("footer");
        return new VBox(footer);
    }

    private VBox metricCard(String caption, Label value) {
        Label heading = new Label(caption);
        heading.getStyleClass().add("metric-caption");
        value.getStyleClass().add("metric-value");
        VBox tile = new VBox(8, heading, value);
        tile.getStyleClass().add("metric-tile");
        tile.setMinWidth(170);
        tile.setPrefWidth(220);
        return tile;
    }

    private VBox card(javafx.scene.Node... nodes) {
        VBox box = new VBox(14, nodes);
        box.getStyleClass().add("surface-card");
        box.setPadding(new Insets(20));
        box.setMinWidth(280);
        box.setMaxWidth(900);
        return box;
    }

    private TableColumn<SampleRow, String> column(
            String title,
            java.util.function.Function<SampleRow, String> value,
            double width
    ) {
        TableColumn<SampleRow, String> column = new TableColumn<>(title);
        column.setPrefWidth(width);
        column.setCellValueFactory(cell -> new ReadOnlyStringWrapper(value.apply(cell.getValue())));
        return column;
    }

    private Region spacer() {
        Region region = new Region();
        HBox.setHgrow(region, Priority.ALWAYS);
        return region;
    }

    private ApiClient client() {
        return new ApiClient(serviceAddress.getText());
    }

    private void connect() {
        setBusy(connectButton, true, "Checking service...");
        requests.submit(() -> {
            try {
                JsonObject response = client().health();
                Platform.runLater(() -> {
                    connectionStatus.setText("Engine online");
                    connectionStatus.getStyleClass().setAll("connection-online");
                    boolean modelAvailable = response.get("model_available").getAsBoolean();
                    modelStatus.setText(modelAvailable ? "Saved model ready" : "No saved model found");
                    startButton.setDisable(!modelAvailable || sessionId != null);
                    feedback.setText("Connected. Training data " +
                            (response.get("dataset_available").getAsBoolean() ? "available." : "not found."));
                    setBusy(connectButton, false, "Reconnect");
                });
            } catch (Exception exception) {
                Platform.runLater(() -> {
                    connectionStatus.setText("Engine offline");
                    connectionStatus.getStyleClass().setAll("connection-offline");
                    modelStatus.setText("Unable to read model status");
                    feedback.setText(exception.getMessage());
                    setBusy(connectButton, false, "Connect");
                });
            }
        });
    }

    private void train() {
        trainButton.setDisable(true);
        feedback.setText("Training from the configured dataset...");
        requests.submit(() -> {
            try {
                client().train();
                JsonObject health = client().health();
                Platform.runLater(() -> {
                    modelStatus.setText("Saved model ready");
                    startButton.setDisable(sessionId != null);
                    feedback.setText("Training finished. The model is ready for simulation.");
                    trainButton.setDisable(false);
                });
            } catch (Exception exception) {
                Platform.runLater(() -> {
                    feedback.setText(exception.getMessage());
                    trainButton.setDisable(false);
                });
            }
        });
    }

    private void startSimulation() {
        int seconds;
        try {
            seconds = Integer.parseInt(duration.getEditor().getText().trim());
            if (seconds < 5 || seconds > 3600) {
                throw new NumberFormatException();
            }
        } catch (NumberFormatException exception) {
            feedback.setText("Choose a whole duration between 5 and 3600 seconds.");
            return;
        }

        startButton.setDisable(true);
        feedback.setText("Starting the simulation...");
        requests.submit(() -> {
            try {
                JsonObject response = client().startSimulation(seconds);
                String newSessionId = response.get("id").getAsString();
                Platform.runLater(() -> {
                    sessionId = newSessionId;
                    sampleIndex = 0;
                    lastRecordedAt = "";
                    recentSamples.clear();
                    scoreSeries.getData().clear();
                    predictionValue.setText("Waiting for first sample");
                    confidenceValue.setText("--");
                    stopButton.setDisable(false);
                    feedback.setText("Simulation running for " + seconds + " seconds.");
                    activity.setVisible(true);
                });
                beginPolling(newSessionId);
            } catch (Exception exception) {
                Platform.runLater(() -> {
                    feedback.setText(exception.getMessage());
                    startButton.setDisable(false);
                });
            }
        });
    }

    private void beginPolling(String activeSessionId) {
        polling.scheduleAtFixedRate(() -> {
            if (!pollInFlight.compareAndSet(false, true)) {
                return;
            }
            requests.submit(() -> {
                try {
                    JsonObject status = client().simulation(activeSessionId);
                    Platform.runLater(() -> updateSimulation(status, activeSessionId));
                } catch (Exception exception) {
                    Platform.runLater(() -> feedback.setText(exception.getMessage()));
                } finally {
                    pollInFlight.set(false);
                }
            });
        }, 0, 1, TimeUnit.SECONDS);
    }

    private void updateSimulation(JsonObject status, String activeSessionId) {
        if (!activeSessionId.equals(sessionId)) {
            return;
        }
        String state = status.get("status").getAsString();
        JsonArray samples = status.getAsJsonArray("samples");
        for (int index = 0; index < samples.size(); index++) {
            JsonObject sample = samples.get(index).getAsJsonObject();
            String recordedAt = sample.has("recorded_at") ? sample.get("recorded_at").getAsString() : "";
            if (!recordedAt.isEmpty() && recordedAt.compareTo(lastRecordedAt) > 0) {
                int sequence = sampleIndex + 1;
                double score = sample.get("fomo_score").getAsDouble();
                int actualLabel = sample.get("real_label").getAsInt();
                int predictedLabel = sample.get("predicted").getAsInt();
                double confidence = sample.get("confidence").getAsDouble();
                String actualName = labelName(actualLabel);
                String predictedName = labelName(predictedLabel);

                predictionValue.setText(predictedName);
                confidenceValue.setText(String.format("%.0f%%", confidence * 100));
                scoreSeries.getData().add(new XYChart.Data<>(sequence, score));
                recentSamples.add(0, new SampleRow(
                        Integer.toString(sequence), actualName, predictedName,
                        String.format("%.3f", score), String.format("%.1f%%", confidence * 100)));
                if (recentSamples.size() > 20) {
                    recentSamples.remove(recentSamples.size() - 1);
                }
                sampleIndex++;
                            lastRecordedAt = recordedAt;
            }
        }
        sampleCount.setText(sampleIndex + (sampleIndex == 1 ? " sample" : " samples"));
        if (state.equals("completed") || state.equals("stopped") || state.equals("failed")) {
            sessionId = null;
            stopButton.setDisable(true);
            startButton.setDisable(false);
            activity.setVisible(false);
            String error = status.has("error") && !status.get("error").isJsonNull()
                    ? status.get("error").getAsString()
                    : null;
            feedback.setText(error == null ? "Simulation " + state + "." : "Simulation failed: " + error);
        } else if (state.equals("stopping")) {
            feedback.setText("Stopping simulation...");
        }
    }

    private String labelName(int label) {
        return switch (label) {
            case 0 -> "Normal";
            case 1 -> "FOMO";
            case 2 -> "Relaxed";
            case 3 -> "Focused";
            default -> "Unknown";
        };
    }

    private void stopSimulation() {
        String activeSessionId = sessionId;
        if (activeSessionId == null) {
            return;
        }
        stopButton.setDisable(true);
        feedback.setText("Stopping simulation...");
        requests.submit(() -> {
            try {
                client().stopSimulation(activeSessionId);
            } catch (IOException exception) {
                Platform.runLater(() -> {
                    feedback.setText(exception.getMessage());
                    stopButton.setDisable(false);
                });
            }
        });
    }

    private void setBusy(Button button, boolean busy, String idleText) {
        button.setDisable(busy);
        button.setText(busy ? "Please wait..." : idleText);
    }

    private void shutdown() {
        polling.shutdownNow();
        requests.shutdownNow();
    }

    public record SampleRow(String index, String actual, String predicted, String score, String confidence) { }

    public static void main(String[] args) {
        launch(args);
    }
}