// Bounded link functions

struct LinkFunctionParams {
    edge: f64,
}

impl LinkFunctionParams {
    fn new(edge: f64) -> Self {
        LinkFunctionParams { edge }
    }
}
