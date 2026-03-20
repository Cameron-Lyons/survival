#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DynamicDeepHitConfig::new(
            TemporalType::LSTM,
            64,
            2,
            false,
            vec![64],
            vec![32],
            10,
            1,
            0.1,
            0.5,
            0.1,
            0.001,
            64,
            100,
            None,
            0.1,
            None,
        )
        .unwrap();

        assert_eq!(config.embedding_dim, 64);
        assert_eq!(config.num_temporal_layers, 2);
    }

    #[test]
    fn test_config_validation() {
        assert!(
            DynamicDeepHitConfig::new(
                TemporalType::LSTM,
                0,
                2,
                false,
                vec![64],
                vec![32],
                10,
                1,
                0.1,
                0.5,
                0.1,
                0.001,
                64,
                100,
                None,
                0.1,
                None
            )
            .is_err()
        );

        assert!(
            DynamicDeepHitConfig::new(
                TemporalType::LSTM,
                64,
                0,
                false,
                vec![64],
                vec![32],
                10,
                1,
                0.1,
                0.5,
                0.1,
                0.001,
                64,
                100,
                None,
                0.1,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn test_softmax_hazards() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, 1.5, 2.5];
        let hazards = softmax_hazards(&logits, 2, 3, 1);

        assert_eq!(hazards.len(), 6);
        for &h in &hazards {
            assert!((0.0..=1.0).contains(&h));
        }
    }
}
