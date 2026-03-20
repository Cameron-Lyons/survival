#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_survival_basic() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 0, 1, 0, 1, 0];
        let covariates = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let adjacency = vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 0.0,
        ];

        let config = NetworkSurvivalConfig::new(true, true, CentralityType::Degree, 1, 50, 1e-4);
        let result =
            network_survival_model(time, event, covariates, 1, adjacency, 6, &config).unwrap();

        assert_eq!(result.n_nodes, 6);
        assert!(result.centrality_values.len() == 6);
    }

    #[test]
    fn test_diffusion_survival() {
        let infection_time = vec![0.0, 1.0, 2.0, 3.0, 10.0, 10.0];
        let infected = vec![1, 1, 1, 1, 0, 0];
        let covariates = vec![1.0, 0.5, 0.8, 0.3, 0.9, 0.2];
        let adjacency = vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 0.0,
        ];

        let config = DiffusionSurvivalConfig::new(0.2, 0.1, true, 50, 1e-4);
        let result = diffusion_survival_model(
            infection_time,
            infected,
            covariates,
            1,
            adjacency,
            6,
            &config,
        )
        .unwrap();

        assert!(result.diffusion_rate.is_finite());
        assert!(result.r0.is_finite());
        assert!(result.infection_probabilities.len() == 6);
    }

    #[test]
    fn test_network_heterogeneity() {
        let time = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = vec![1, 0, 1, 0, 1, 0];
        let adjacency = vec![
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 0.0,
        ];

        let result = network_heterogeneity_survival(time, event, adjacency, 6, Some(2)).unwrap();

        assert_eq!(result.community_assignments.len(), 6);
        assert_eq!(result.community_hazard_ratios.len(), 2);
    }

    #[test]
    fn test_centrality_computations() {
        let adjacency = vec![
            0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ];

        let degree = compute_centrality(&adjacency, 4, &CentralityType::Degree);
        assert_eq!(degree.len(), 4);
        assert!((degree[1] - 3.0).abs() < 1e-6);

        let pagerank = compute_centrality(&adjacency, 4, &CentralityType::PageRank);
        assert_eq!(pagerank.len(), 4);
        assert!(pagerank.iter().all(|&p| p > 0.0 && p < 1.0));
    }
}
