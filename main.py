from components import ParticleClassifier, ParticleDataPreprocessor, ParticleVisualizer
from data.DataDownloader import ParticleDataDownloader

def compare_model_metrics(models_dict, test_dataset):
    results = {}
    for name, classifier in models_dict.items():
        metrics = classifier.evaluate(test_dataset)
        results[name] = metrics
        print(f"\n{name} Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    return results

if __name__ == "__main__":
    #Download data
    ParticleDataDownloader()

    # Initialize preprocessor
    preprocessor = ParticleDataPreprocessor()
    
    # Preprocess data
    train_dataset, test_dataset, input_shape = preprocessor.preprocess_data(
        electrons_path='./data/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5',
        photons_path='./data/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5',
        sample_fraction=0.2,
        batch_size=32
    )

    # Validate the data
    preprocessor.validate_data(train_dataset)

    # lenet5 initialization
    classifier_lenet = ParticleClassifier(model_type='lenet5')

    # Train the model
    classifier_lenet.train(train_dataset, test_dataset)
    classifier_lenet.evaluate(test_dataset)

    classifier_lenet.summary_model()

    classifier_lenet.plot_history()


    classifier_lenet.load_model('lenet5.keras')

    visualizer_lenet = ParticleVisualizer(classifier_lenet.model)

    print("\nAnalyzing LeNet-5 performance...")
    pred_lenet, true_lenet = visualizer_lenet.analyze_model_performance(test_dataset)

    models = {
        'LeNet-5': classifier_lenet
    }

    metrics_comparison = compare_model_metrics(models, test_dataset)

