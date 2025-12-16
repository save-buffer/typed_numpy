use pyo3::prelude::*;

#[allow(nonstandard_style)]
#[pymodule]
mod typed_numpy
{
    use egglog::ast::Schema;
    use egglog::prelude::*;
    use egglog::add_primitive;
    use egglog::sort::{F, OrderedFloat};
    use pyo3::prelude::*;

    #[derive(Debug, Clone)]
    struct TypedNumpyError
    {
        message : String,
    }

    impl From<egglog::Error> for TypedNumpyError
    {
        fn from(err : egglog::Error) -> TypedNumpyError
        {
            return TypedNumpyError
            {
                message : err.to_string(),
            };
        }
    }

    impl From<TypedNumpyError> for PyErr
    {
        fn from(err : TypedNumpyError) -> PyErr
        {
            return pyo3::exceptions::PyRuntimeError::new_err(err.message);
        }
    }

    #[pyclass]
    struct RustEgraph
    {
        eg : egglog::EGraph,
    }

    fn NewRustEgraph() -> Result<egglog::EGraph, egglog::Error>
    {
        let mut eg = egglog::EGraph::default();
        datatype!(&mut eg,
            (datatype Expr
                (Constant f64)
                (Tensor)
                (Exp Expr)
                (Sin Expr)
                (Cos Expr)
                (Sqrt Expr)
                (Add Expr Expr)
                (Sub Expr Expr)
                (Mul Expr Expr)
                (Div Expr Expr)
                (BinaryMax Expr Expr)
                (Repeat String Expr)
                (Sum String i64 i64 Expr)
                (Max String i64 i64 Expr)
            )
        );

        add_primitive!(&mut eg, "exp" = |a: F| -> F { F::from(OrderedFloat((**a).exp())) });
        add_primitive!(&mut eg, "sin" = |a: F| -> F { F::from(OrderedFloat((**a).sin())) });
        add_primitive!(&mut eg, "cos" = |a: F| -> F { F::from(OrderedFloat((**a).cos())) });
        add_primitive!(&mut eg, "sqrt" = |a: F| -> F { F::from(OrderedFloat((**a).sqrt())) });

        let ruleset = "rewrites";
        add_ruleset(&mut eg, ruleset)?;

        macro_rules! add_rule
        {
            ($lhs:tt => $rhs:tt) =>
            {
                rule(&mut eg, ruleset, facts![(= x $lhs)], actions![(union x $rhs)])?;
            };
            ($lhs:tt <=> $rhs:tt) =>
            {
                add_rule!($lhs => $rhs);
                add_rule!($rhs => $lhs);
            };
        }


        add_rule!((Add a b) => (Add b a)); // commutativity
        add_rule!((Add (Add a b) c) <=> (Add a (Add b c))); // associativity
        add_rule!((Mul a (Add b c)) <=> (Add (Mul a b) (Mul a c))); // distributivity
        add_rule!((Div (Add a b) c) <=> (Add (Div a c) (Div b c))); // div_distributivity
        // repeat_over_unary_ops
        add_rule!((Exp (Repeat D a)) <=> (Repeat D (Exp a)));
        add_rule!((Sin (Repeat D a)) <=> (Repeat D (Sin a)));
        add_rule!((Cos (Repeat D a)) <=> (Repeat D (Cos a)));
        add_rule!((Sqrt (Repeat D a)) <=> (Repeat D (Sqrt a)));
        // repeat_over_binary_ops
        add_rule!((Add (Repeat D a) (Repeat D b)) <=> (Repeat D (Add a b)));
        add_rule!((Sub (Repeat D a) (Repeat D b)) <=> (Repeat D (Sub a b)));
        add_rule!((Mul (Repeat D a) (Repeat D b)) <=> (Repeat D (Mul a b)));
        add_rule!((Div (Repeat D a) (Repeat D b)) <=> (Repeat D (Div a b)));
        add_rule!((BinaryMax (Repeat D a) (Repeat D b)) <=> (Repeat D (BinaryMax a b)));
        // reorder_reductions
        add_rule!((Sum D s e (Sum E m n a)) => (Sum E m n (Sum D s e a)));
        add_rule!((Max D s e (Max E m n a)) => (Max E m n (Max D s e a)));

        add_rule!((Repeat D (Repeat E x)) => (Repeat E (Repeat D x))); // reorder_repeats
        add_rule!((Sum D s e (Div a (Repeat D b))) <=> (Div (Sum D s e a) b)); // factor_out_of_sum
        add_rule!((Mul x (Constant (unquote exprs::float(0.0)))) => (Constant (unquote exprs::float(0.0)))); // mul_by_zero
        add_rule!((Div a (Div b c)) <=> (Mul a (Div c b))); // div_of_quotient_to_mul_of_reciprocal
        add_rule!((Add (Div a d) (Div b d)) <=> (Div (Add a b) d)); // add_fractions
        add_rule!((Sub (Div a d) (Div b d)) <=> (Div (Sub a b) d)); // sub_fractions
        add_rule!((Mul (Div a c) (Div b d)) <=> (Div (Mul a b) (Mul c d))); // multiply_fractions
        add_rule!((Div (Mul a b) (Mul c d)) <=> (Mul (Div a c) (Div b d))); // decompose_fractions
        add_rule!((Sub a a) => (Constant (unquote exprs::float(0.0)))); // sub_by_self
        add_rule!((Div a a) => (Constant (unquote exprs::float(1.0)))); // div_by_self
        add_rule!((Add a (Constant (unquote exprs::float(0.0)))) => a); // identity_add
        add_rule!((Sub a (Constant (unquote exprs::float(0.0)))) => a); // identity_sub
        add_rule!((Mul a (Constant (unquote exprs::float(1.0)))) => a); // identity_mul
        add_rule!((Div a (Constant (unquote exprs::float(1.0)))) => a); // identity_div
        add_rule!((Exp (Constant a)) => (Constant (exp a))); // constant_folding_exp
        add_rule!((Sin (Constant a)) => (Constant (sin a))); // constant_folding_sin
        add_rule!((Cos (Constant a)) => (Constant (cos a))); // constant_folding_cos
        add_rule!((Sqrt (Constant a)) => (Constant (sqrt a))); // constant_folding_sqrt
        add_rule!((Add (Constant a) (Constant b)) => (Constant (+ a b))); // constant_folding_add
        add_rule!((Sub (Constant a) (Constant b)) => (Constant (- a b))); // constant_folding_sub
        add_rule!((Mul (Constant a) (Constant b)) => (Constant (* a b))); // constant_folding_mul
        add_rule!((Div (Constant a) (Constant b)) => (Constant (/ a b))); // constant_folding_div
        add_rule!((Exp (Add x y)) <=> (Mul (Exp x) (Exp y))); // product_of_exp
        add_rule!((Exp (Sub x y)) <=> (Div (Exp x) (Exp y))); // quotient_of_exp

        // log_sum_exp_stability
        add_rule!((Sum D s e (Exp x)) <=>
            (Mul
                (Exp (Max D s e x))
                (Sum D s e
                    (Exp
                        (Sub x
                            (Repeat D
                                (Max D s e x)))))));

        // numerically_stable_softmax
        add_rule!((Div (Exp x) (Repeat D (Sum D s e (Exp x)))) <=>
            (Div (Exp (Sub x (Repeat D (Max D s e x))))
                (Repeat D (Sum D s e (Exp (Sub x (Repeat D (Max D s e x))))))));

        return Ok(eg);
    }

    #[pymethods]
    impl RustEgraph
    {
        #[new]
        fn py_new() -> PyResult<Self>
        {
            println!("Making new egraph!");
            let eg = NewRustEgraph().map_err(TypedNumpyError::from)?;
            return Ok(
                RustEgraph
                {
                    eg : eg
                }
            );
        }
    }
}
