//! parser combinators
//!
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub struct ParseError {
    // number of characters finished reading at the time the error occurred
    position: usize,
    // expected tokens
    expected: Vec<String>,
    // subsequent actually found
    found: Option<String>,
}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ParseError: position {}, expected: {}, found: {}",
            self.position,
            self.expected.join(" or "),
            self.found.clone().unwrap_or_else(|| "EOS".to_string())
        )
    }
}
impl ParseError {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn expected(&self) -> &Vec<String> {
        &self.expected
    }
    pub fn found(&self) -> Option<&String> {
        self.found.as_ref()
    }
}

pub type ParseResult<'a, T> = Result<(T, &'a str), ParseError>;

pub trait Parser: Clone {
    type Item;

    fn parse(self, input: &str) -> ParseResult<Self::Item>;

    fn label(self, label: String) -> impl Parser<Item = Self::Item>
    where
        Self: Sized,
    {
        Label {
            parser: self,
            label,
        }
    }

    fn map<F, T>(self, f: F) -> impl Parser<Item = T>
    where
        Self: Sized,
        F: FnOnce(Self::Item) -> T + Clone,
    {
        Map { parser: self, f }
    }
    fn opt(self) -> impl Parser<Item = Option<Self::Item>>
    where
        Self: Sized,
    {
        Opt { parser: self }
    }
    fn join<Q>(self, parser2: Q) -> impl Parser<Item = (Self::Item, Q::Item)>
    where
        Self: Sized,
        Q: Parser,
    {
        Join {
            parser1: self,
            parser2,
        }
    }
    fn with<Q>(self, parser2: Q) -> impl Parser<Item = Self::Item>
    where
        Self: Sized,
        Q: Parser,
    {
        With {
            parser1: self,
            parser2,
        }
    }
    fn skip<Q>(self, parser2: Q) -> impl Parser<Item = Q::Item>
    where
        Self: Sized,
        Q: Parser,
    {
        Skip {
            parser1: self,
            parser2,
        }
    }
    fn and_then<F, Q>(self, f: F) -> impl Parser<Item = Q::Item>
    where
        Self: Sized,
        F: FnOnce(Self::Item) -> Q + Clone,
        Q: Parser,
    {
        AndThen { parser: self, f }
    }
    fn or<Q>(self, parser2: Q) -> impl Parser<Item = Self::Item>
    where
        Self: Sized,
        Q: Parser<Item = Self::Item>,
    {
        Or {
            parser1: self,
            parser2,
        }
    }
    fn many1(self) -> impl Parser<Item = Vec<Self::Item>>
    where
        Self: Sized + Clone,
    {
        Many1 { parser: self }
    }
    fn many0(self) -> impl Parser<Item = Vec<Self::Item>>
    where
        Self: Sized + Clone,
    {
        Many0 { parser: self }
    }
    fn sep_by<Q>(self, sep: Q) -> impl Parser<Item = Vec<Self::Item>>
    where
        Self: Sized + Clone,
        Q: Parser,
    {
        SepBy { parser: self, sep }
    }
    fn bracket<P, Q>(self, open: P, close: Q) -> impl Parser<Item = Self::Item>
    where
        Self: Sized + Clone,
        P: Parser,
        Q: Parser,
    {
        Bracket {
            open,
            parser: self,
            close,
        }
    }
}

impl<F, T> Parser for F
where
    F: FnOnce(&str) -> ParseResult<T> + Clone,
{
    type Item = T;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        self(input)
    }
}

fn _label<P>(parser: P, label: String) -> impl FnOnce(&str) -> ParseResult<P::Item>
where
    P: Parser,
{
    move |input| match parser.parse(input) {
        Ok(x) => Ok(x),
        Err(mut err) => {
            err.expected.pop();
            err.expected.push(label);
            Err(err)
        }
    }
}
#[derive(Debug, Clone)]
pub struct Label<P> {
    parser: P,
    label: String,
}
impl<P> Parser for Label<P>
where
    P: Parser,
{
    type Item = P::Item;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _label(self.parser, self.label)(input)
    }
}

fn _map<P, F, T>(parser: P, f: F) -> impl FnOnce(&str) -> ParseResult<T>
where
    P: Parser,
    F: FnOnce(P::Item) -> T,
{
    move |input| match parser.parse(input) {
        Ok((x, rest)) => Ok((f(x), rest)),
        Err(e) => Err(e),
    }
}
#[derive(Debug, Clone)]
pub struct Map<P, F> {
    parser: P,
    f: F,
}
impl<P, F, T> Parser for Map<P, F>
where
    P: Parser,
    F: FnOnce(P::Item) -> T + Clone,
{
    type Item = T;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _map(self.parser, self.f)(input)
    }
}
#[cfg(test)]
mod test_map {
    use super::*;

    #[test]
    fn test_map() {
        assert_eq!(int32().map(|x| x * 2).parse("123abc"), Ok((246, "abc")));
        assert_eq!(
            int32().map(|x| x * 2).parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

fn _opt<P>(parser: P) -> impl FnOnce(&str) -> ParseResult<Option<P::Item>>
where
    P: Parser,
{
    move |input| match parser.parse(input) {
        Ok((x, rest)) => Ok((Some(x), rest)),
        Err(_) => Ok((None, input)),
    }
}
#[derive(Debug, Clone)]
pub struct Opt<P> {
    parser: P,
}
impl<P> Parser for Opt<P>
where
    P: Parser,
{
    type Item = Option<P::Item>;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _opt(self.parser)(input)
    }
}
#[cfg(test)]
mod test_opt {
    use super::*;

    #[test]
    fn test_opt() {
        assert_eq!(int32().opt().parse("123abc"), Ok((Some(123), "abc")));
        assert_eq!(int32().opt().parse("abc"), Ok((None, "abc")));
    }
}

fn _join<P, Q>(parser1: P, parser2: Q) -> impl FnOnce(&str) -> ParseResult<(P::Item, Q::Item)>
where
    P: Parser,
    Q: Parser,
{
    move |input| match parser1.parse(input) {
        Ok((x, rest)) => match parser2.parse(rest) {
            Ok((y, rest)) => Ok(((x, y), rest)),
            Err(mut e) => {
                e.position += input.len() - rest.len();
                Err(e)
            }
        },
        Err(e) => Err(e),
    }
}
#[derive(Debug, Clone)]
pub struct Join<P, Q> {
    parser1: P,
    parser2: Q,
}
impl<P, Q> Parser for Join<P, Q>
where
    P: Parser,
    Q: Parser,
{
    type Item = (P::Item, Q::Item);

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _join(self.parser1, self.parser2)(input)
    }
}
#[cfg(test)]
mod test_join {
    use super::*;

    #[test]
    fn test_join() {
        assert_eq!(
            int32().join(char('a')).parse("123abc"),
            Ok(((123, 'a'), "bc"))
        );
        assert_eq!(
            int32().join(char('a')).parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
        assert_eq!(
            int32().join(char('a')).parse("123xyz"),
            Err(ParseError {
                position: 3,
                expected: vec!["char 'a'".to_string()],
                found: Some("x".to_string())
            })
        );
    }
}

fn _with<P, Q>(parser1: P, parser2: Q) -> impl FnOnce(&str) -> ParseResult<P::Item>
where
    P: Parser,
    Q: Parser,
{
    move |input| match parser1.parse(input) {
        Ok((x, rest1)) => match parser2.parse(rest1) {
            Ok((_, rest2)) => Ok((x, rest2)),
            Err(mut e) => {
                e.position += input.len() - rest1.len();
                Err(e)
            }
        },
        Err(e) => Err(e),
    }
}
#[derive(Debug, Clone)]
pub struct With<P, Q> {
    parser1: P,
    parser2: Q,
}
impl<P, Q> Parser for With<P, Q>
where
    P: Parser,
    Q: Parser,
{
    type Item = P::Item;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _with(self.parser1, self.parser2)(input)
    }
}
#[cfg(test)]
mod test_with {
    use super::*;

    #[test]
    fn test_with() {
        assert_eq!(int32().with(char('a')).parse("123abc"), Ok((123, "bc")));
        assert_eq!(
            int32().with(char('a')).parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
        assert_eq!(
            int32().with(char('a')).parse("123bc"),
            Err(ParseError {
                position: 3,
                expected: vec!["char 'a'".to_string()],
                found: Some("b".to_string())
            })
        );
    }
}

fn _skip<P, Q>(parser1: P, parser2: Q) -> impl FnOnce(&str) -> ParseResult<Q::Item>
where
    P: Parser,
    Q: Parser,
{
    move |input| match parser1.parse(input) {
        Ok((_, rest1)) => match parser2.parse(rest1) {
            Ok((x, rest2)) => Ok((x, rest2)),
            Err(mut e) => {
                e.position += input.len() - rest1.len();
                Err(e)
            }
        },
        Err(e) => Err(e),
    }
}
#[derive(Debug, Clone)]
pub struct Skip<P, Q> {
    parser1: P,
    parser2: Q,
}
impl<P, Q> Parser for Skip<P, Q>
where
    P: Parser,
    Q: Parser,
{
    type Item = Q::Item;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _skip(self.parser1, self.parser2)(input)
    }
}
#[cfg(test)]
mod test_skip {
    use super::*;

    #[test]
    fn test_skip() {
        assert_eq!(int32().skip(char('a')).parse("123abc"), Ok(('a', "bc")));
        assert_eq!(
            int32().skip(char('a')).parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
        assert_eq!(
            int32().skip(char('a')).parse("123bc"),
            Err(ParseError {
                position: 3,
                expected: vec!["char 'a'".to_string()],
                found: Some("b".to_string())
            })
        );
    }
}

fn _and_then<P, Q, F>(parser: P, f: F) -> impl FnOnce(&str) -> ParseResult<Q::Item>
where
    P: Parser,
    Q: Parser,
    F: FnOnce(P::Item) -> Q,
{
    move |input| match parser.parse(input) {
        Ok((x, rest1)) => match f(x).parse(rest1) {
            Ok((y, rest2)) => Ok((y, rest2)),
            Err(mut e) => {
                e.position += input.len() - rest1.len();
                Err(e)
            }
        },
        Err(e) => Err(e),
    }
}
#[derive(Debug, Clone)]
pub struct AndThen<P, F> {
    parser: P,
    f: F,
}
impl<P, Q, F> Parser for AndThen<P, F>
where
    P: Parser,
    Q: Parser,
    F: FnOnce(P::Item) -> Q + Clone,
{
    type Item = Q::Item;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _and_then(self.parser, self.f)(input)
    }
}
#[cfg(test)]
mod test_and_then {
    use super::*;

    #[test]
    fn test_and_then() {
        assert_eq!(
            int32()
                .and_then(|x| char('a').map(move |y| (x, y)))
                .parse("123abc"),
            Ok(((123, 'a'), "bc"))
        );
        assert_eq!(
            int32()
                .and_then(|x| char('a').map(move |y| (x, y)))
                .parse("123xyz"),
            Err(ParseError {
                position: 3,
                expected: vec!["char 'a'".to_string()],
                found: Some("x".to_string())
            })
        );
        assert_eq!(
            int32()
                .and_then(|x| char('a').map(move |y| (x, y)))
                .parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

fn _or<P, Q>(parser1: P, parser2: Q) -> impl FnOnce(&str) -> ParseResult<P::Item>
where
    P: Parser,
    Q: Parser<Item = P::Item>,
{
    move |input| match parser1.parse(input) {
        Ok((x, rest)) => Ok((x, rest)),
        Err(mut e1) => match parser2.parse(input) {
            Ok((x, rest)) => Ok((x, rest)),
            Err(e2) => {
                e1.expected.extend(e2.expected);
                Err(e1)
            }
        },
    }
}
#[derive(Debug, Clone)]
pub struct Or<P, Q> {
    parser1: P,
    parser2: Q,
}
impl<P, Q> Parser for Or<P, Q>
where
    P: Parser,
    Q: Parser<Item = P::Item>,
{
    type Item = P::Item;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _or(self.parser1, self.parser2)(input)
    }
}
#[cfg(test)]
mod test_or {
    use super::*;

    #[test]
    fn test_or() {
        assert_eq!(char('a').or(char('A')).parse("abc"), Ok(('a', "bc")));
        assert_eq!(char('a').or(char('A')).parse("ABC"), Ok(('A', "BC")));
        assert_eq!(
            keyword("abc").or(keyword("aBC")).parse("aBCdef"),
            Ok(("aBC", "def"))
        );
        assert_eq!(
            char('a')
                .join(char('b'))
                .or(char('a').join(char('B')))
                .parse("aBCdef"),
            Ok((('a', 'B'), "Cdef"))
        );
        assert_eq!(
            char('a').or(char('A')).parse("123"),
            Err(ParseError {
                position: 0,
                expected: vec!["char 'a'".to_string(), "char 'A'".to_string()],
                found: Some("1".to_string())
            })
        );
    }
}

fn _many1<P>(parser: P) -> impl FnOnce(&str) -> ParseResult<Vec<P::Item>>
where
    P: Parser + Clone,
{
    move |input| {
        let mut result = Vec::new();

        match parser.clone().parse(input) {
            Ok((x, mut rest)) => {
                result.push(x);
                loop {
                    match parser.clone().parse(rest) {
                        Ok((x, r)) => {
                            rest = r;
                            result.push(x);
                        }
                        Err(_) => return Ok((result, rest)),
                    }
                }
            }
            Err(e) => Err(e),
        }
    }
}
#[derive(Debug, Clone)]
pub struct Many1<P> {
    parser: P,
}
impl<P> Parser for Many1<P>
where
    P: Parser + Clone,
{
    type Item = Vec<P::Item>;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _many1(self.parser)(input)
    }
}
#[cfg(test)]
mod test_many1 {
    use super::*;

    #[test]
    fn test_many1() {
        assert_eq!(char('a').many1().parse("abc"), Ok((vec!['a'], "bc")));
        assert_eq!(char('a').many1().parse("aabc"), Ok((vec!['a', 'a'], "bc")));
        assert_eq!(
            char('a').many1().parse("123"),
            Err(ParseError {
                position: 0,
                expected: vec!["char 'a'".to_string()],
                found: Some("1".to_string())
            })
        );
    }
}

fn _many0<P>(parser: P) -> impl FnOnce(&str) -> ParseResult<Vec<P::Item>>
where
    P: Parser + Clone,
{
    move |input| {
        let mut result = Vec::new();
        let mut rest = input;

        loop {
            match parser.clone().parse(rest) {
                Ok((x, r)) => {
                    rest = r;
                    result.push(x);
                }
                Err(_) => return Ok((result, rest)),
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct Many0<P> {
    parser: P,
}
impl<P> Parser for Many0<P>
where
    P: Parser + Clone,
{
    type Item = Vec<P::Item>;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _many0(self.parser)(input)
    }
}
#[cfg(test)]
mod test_many0 {
    use super::*;

    #[test]
    fn test_many0() {
        assert_eq!(char('a').many0().parse("abc"), Ok((vec!['a'], "bc")));
        assert_eq!(char('a').many0().parse("aabc"), Ok((vec!['a', 'a'], "bc")));
        assert_eq!(char('a').many0().parse("123"), Ok((vec![], "123")));
    }
}

fn _sep_by<P, Q>(parser: P, sep: Q) -> impl FnOnce(&str) -> ParseResult<Vec<P::Item>>
where
    P: Parser + Clone,
    Q: Parser,
{
    move |input| {
        let mut result = Vec::new();
        let mut rest = input;

        match parser.clone().parse(rest) {
            Ok((x, r)) => {
                rest = r;
                result.push(x);
                loop {
                    match sep.clone().parse(rest) {
                        Ok((_, r)) => {
                            rest = r;
                            match parser.clone().parse(rest) {
                                Ok((x, r)) => {
                                    rest = r;
                                    result.push(x);
                                }
                                Err(_) => return Ok((result, rest)),
                            }
                        }
                        Err(_) => return Ok((result, rest)),
                    }
                }
            }
            Err(mut e) => {
                e.position += input.len() - rest.len();
                Err(e)
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct SepBy<P, Q> {
    parser: P,
    sep: Q,
}
impl<P, Q> Parser for SepBy<P, Q>
where
    P: Parser + Clone,
    Q: Parser,
{
    type Item = Vec<P::Item>;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _sep_by(self.parser, self.sep)(input)
    }
}
#[cfg(test)]
mod test_sep_by {
    use super::*;

    #[test]
    fn test_sep_by() {
        assert_eq!(
            int32().sep_by(char(',')).parse("123,456,789abc"),
            Ok((vec![123, 456, 789], "abc"))
        );
        assert_eq!(
            int32().sep_by(char(',')).parse("abc,def,ghi,"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

fn _bracket<P, Q, R>(open: P, parser: Q, close: R) -> impl FnOnce(&str) -> ParseResult<Q::Item>
where
    P: Parser,
    Q: Parser,
    R: Parser,
{
    move |input| match open.parse(input) {
        Err(e1) => return Err(e1),
        Ok((_, rest1)) => match parser.parse(rest1) {
            Err(mut e2) => {
                e2.position += input.len() - rest1.len();
                return Err(e2);
            }
            Ok((x, rest2)) => match close.parse(rest2) {
                Err(mut e3) => {
                    e3.position += input.len() - rest2.len();
                    return Err(e3);
                }
                Ok((_, rest3)) => Ok((x, rest3)),
            },
        },
    }
}
#[derive(Debug, Clone)]
pub struct Bracket<P, Q, R> {
    open: P,
    parser: Q,
    close: R,
}
impl<P, Q, R> Parser for Bracket<P, Q, R>
where
    P: Parser,
    Q: Parser + Clone,
    R: Parser,
{
    type Item = Q::Item;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _bracket(self.open, self.parser, self.close)(input)
    }
}
#[cfg(test)]
mod test_bracket {
    use super::*;

    #[test]
    fn test_bracket() {
        assert_eq!(
            int32().bracket(char('['), char(']')).parse("[123]abc"),
            Ok((123, "abc"))
        );
        assert_eq!(
            int32().bracket(char('"'), char('"')).parse("\"123\"abc"),
            Ok((123, "abc"))
        );
        assert_eq!(
            int32().bracket(char('<'), char('>')).parse("<123abc"),
            Err(ParseError {
                position: 4,
                expected: vec!["char '>'".to_string()],
                found: Some("a".to_string()),
            })
        );
    }
}

pub fn pred<F>(f: F) -> impl Parser<Item = char>
where
    F: FnOnce(char) -> bool + Clone,
{
    Pred { f }
}
fn _pred<F>(f: F) -> impl FnOnce(&str) -> ParseResult<char>
where
    F: FnOnce(char) -> bool,
{
    move |input| match input.chars().next() {
        Some(c) if f(c) => Ok((c, &input[c.len_utf8()..])),
        Some(c) => Err(ParseError {
            position: 0,
            expected: vec![],
            found: Some(c.to_string()),
        }),
        _ => Err(ParseError {
            position: 0,
            expected: vec!["found EOS".to_string()],
            found: None,
        }),
    }
}
#[derive(Debug, Clone)]
pub struct Pred<F> {
    f: F,
}
impl<F> Parser for Pred<F>
where
    F: FnOnce(char) -> bool + Clone,
{
    type Item = char;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _pred(self.f)(input)
    }
}
#[cfg(test)]
mod test_pred {
    use super::*;

    #[test]
    fn test_pred() {
        assert_eq!(pred(|c| c == 'a').parse("abc"), Ok(('a', "bc")));
        assert_eq!(pred(|c| c == '1').parse("123"), Ok(('1', "23")));
        assert_eq!(pred(|c| c == 'あ').parse("あいう"), Ok(('あ', "いう")));
        assert_eq!(
            pred(|c| c == 'a').parse("123"),
            Err(ParseError {
                position: 0,
                expected: vec![],
                found: Some("1".to_string())
            })
        );
        assert_eq!(
            pred(|c| c == 'a').parse(""),
            Err(ParseError {
                position: 0,
                expected: vec!["found EOS".to_string()],
                found: None,
            })
        );
    }
}

pub fn alpha() -> impl Parser<Item = char> {
    pred(|c: char| c.is_alphabetic()).label("alpha".to_string())
}
#[cfg(test)]
mod test_alpha {
    use super::*;

    #[test]
    fn test_alpha() {
        assert_eq!(alpha().parse("abc"), Ok(('a', "bc")));
        assert_eq!(
            alpha().parse("123"),
            Err(ParseError {
                position: 0,
                expected: vec!["alpha".to_string()],
                found: Some("1".to_string())
            })
        );
        assert_eq!(
            alpha().parse(""),
            Err(ParseError {
                position: 0,
                expected: vec!["alpha".to_string()],
                found: None
            })
        );
    }
}

pub fn digit() -> impl Parser<Item = char> {
    pred(|c: char| c.is_digit(10)).label("digit".to_string())
}
#[cfg(test)]
mod test_digit {
    use super::*;

    #[test]
    fn test_digit() {
        assert_eq!(digit().parse("123"), Ok(('1', "23")));
        assert_eq!(
            digit().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn digits() -> impl Parser<Item = Vec<char>> {
    digit().many1()
}
#[cfg(test)]
mod test_digits {
    use super::*;

    #[test]
    fn test_digits() {
        assert_eq!(digits().parse("123"), Ok((vec!['1', '2', '3'], "")));
        assert_eq!(
            digits().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
        assert_eq!(
            digits().parse(""),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: None,
            })
        );
    }
}

pub fn int8() -> impl Parser<Item = i8> {
    let sign = pred(|c| c == '-' || c == '+');
    let digits = digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as i8)
    });

    sign.opt()
        .join(digits)
        .map(|(s, d)| if s == Some('-') { -d } else { d })
}
#[cfg(test)]
mod test_int8 {
    use super::*;

    #[test]
    fn test_int8() {
        assert_eq!(int8().parse("123"), Ok((123, "")));
        assert_eq!(int8().parse("-123"), Ok((-123, "")));
        assert_eq!(int8().parse("+123"), Ok((123, "")));
        assert_eq!(
            int8().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn uint8() -> impl Parser<Item = u8> {
    digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as u8)
    })
}
#[cfg(test)]
mod test_uint8 {
    use super::*;

    #[test]
    fn test_uint8() {
        assert_eq!(uint8().parse("123"), Ok((123, "")));
        assert_eq!(
            uint8().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn int16() -> impl Parser<Item = i16> {
    let sign = pred(|c| c == '-' || c == '+');
    let digits = digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as i16)
    });

    sign.opt()
        .join(digits)
        .map(|(s, d)| if s == Some('-') { -d } else { d })
}
#[cfg(test)]
mod test_int16 {
    use super::*;

    #[test]
    fn test_int16() {
        assert_eq!(int16().parse("123"), Ok((123, "")));
        assert_eq!(int16().parse("-123"), Ok((-123, "")));
        assert_eq!(int16().parse("+123"), Ok((123, "")));
        assert_eq!(
            int16().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn uint16() -> impl Parser<Item = u16> {
    digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as u16)
    })
}
#[cfg(test)]
mod test_uint16 {
    use super::*;

    #[test]
    fn test_uint16() {
        assert_eq!(uint16().parse("123"), Ok((123, "")));
        assert_eq!(
            uint16().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn int32() -> impl Parser<Item = i32> {
    let sign = pred(|c| c == '-' || c == '+');
    let digits = digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as i32)
    });

    sign.opt()
        .join(digits)
        .map(|(s, d)| if s == Some('-') { -d } else { d })
}
#[cfg(test)]
mod test_int32 {
    use super::*;

    #[test]
    fn test_int32() {
        assert_eq!(int32().parse("123"), Ok((123, "")));
        assert_eq!(int32().parse("-123"), Ok((-123, "")));
        assert_eq!(int32().parse("+123"), Ok((123, "")));
        assert_eq!(
            int32().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn uint32() -> impl Parser<Item = u32> {
    digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as u32)
    })
}
#[cfg(test)]
mod test_uint32 {
    use super::*;

    #[test]
    fn test_uint32() {
        assert_eq!(uint32().parse("123"), Ok((123, "")));
        assert_eq!(
            uint32().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn int64() -> impl Parser<Item = i64> {
    let sign = pred(|c| c == '-' || c == '+');
    let digits = digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as i64)
    });

    sign.opt()
        .join(digits)
        .map(|(s, d)| if s == Some('-') { -d } else { d })
}
#[cfg(test)]
mod test_int64 {
    use super::*;

    #[test]
    fn test_int64() {
        assert_eq!(int64().parse("123"), Ok((123, "")));
        assert_eq!(int64().parse("-123"), Ok((-123, "")));
        assert_eq!(int64().parse("+123"), Ok((123, "")));
        assert_eq!(
            int64().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn uint64() -> impl Parser<Item = u64> {
    digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as u64)
    })
}
#[cfg(test)]
mod test_uint64 {
    use super::*;

    #[test]
    fn test_uint64() {
        assert_eq!(uint64().parse("123"), Ok((123, "")));
        assert_eq!(
            uint64().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["digit".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn float32() -> impl Parser<Item = f32> {
    let sign = pred(|c| c == '-' || c == '+');
    let integer = digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as i32)
    });
    let decimal = digits().map(|chars| {
        chars.iter().rev().fold(0.0, |acc, &c| {
            (acc as f32 + c.to_digit(10).unwrap() as f32) / 10.0
        })
    });

    sign.opt()
        .join(integer.opt())
        .with(char('.'))
        .join(decimal)
        .map(|((s, i), d)| {
            let f = i.unwrap_or_default() as f32 + d as f32;
            if s == Some('-') {
                -f
            } else {
                f
            }
        })
}
#[cfg(test)]
mod test_float32 {
    use super::*;

    #[test]
    fn test_float32() {
        assert_eq!(float32().parse("123.456"), Ok((123.456, "")));
        assert_eq!(float32().parse("-123.456"), Ok((-123.456, "")));
        assert_eq!(float32().parse("+123.456"), Ok((123.456, "")));
        assert_eq!(float32().parse(".456"), Ok((0.456, "")));
        assert_eq!(
            float32().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["char '.'".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn float64() -> impl Parser<Item = f64> {
    let sign = pred(|c| c == '-' || c == '+');
    let integer = digits().map(|chars| {
        chars
            .iter()
            .fold(0, |acc, &c| acc * 10 + c.to_digit(10).unwrap() as i64)
    });
    let decimal = digits().map(|chars| {
        chars.iter().rev().fold(0.0, |acc, &c| {
            (acc as f64 + c.to_digit(10).unwrap() as f64) / 10.0
        })
    });

    sign.opt()
        .join(integer.opt())
        .with(char('.'))
        .join(decimal)
        .map(|((s, i), d)| {
            let f = i.unwrap_or_default() as f64 + d as f64;
            if s == Some('-') {
                -f
            } else {
                f
            }
        })
}
#[cfg(test)]
mod test_float64 {
    use super::*;

    #[test]
    fn test_float64() {
        assert_eq!(float32().parse("123.456"), Ok((123.456, "")));
        assert_eq!(float32().parse("-123.456"), Ok((-123.456, "")));
        assert_eq!(float32().parse("+123.456"), Ok((123.456, "")));
        assert_eq!(float32().parse(".456"), Ok((0.456, "")));
        assert_eq!(
            float32().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["char '.'".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn alphanum() -> impl Parser<Item = char> {
    pred(|c: char| c.is_alphanumeric()).label("alphanum".to_string())
}
#[cfg(test)]
mod test_alphanum {
    use super::*;

    #[test]
    fn test_alphanum() {
        assert_eq!(alphanum().parse("abc"), Ok(('a', "bc")));
        assert_eq!(alphanum().parse("123"), Ok(('1', "23")));
        assert_eq!(
            alphanum().parse("+-*/"),
            Err(ParseError {
                position: 0,
                expected: vec!["alphanum".to_string()],
                found: Some("+".to_string())
            })
        );
    }
}

pub fn any_char() -> impl Parser<Item = char> {
    pred(|_| true).label("any char".to_string())
}
#[cfg(test)]
mod test_any_char {
    use super::*;

    #[test]
    fn test_any_char() {
        assert_eq!(any_char().parse("abc"), Ok(('a', "bc")));
        assert_eq!(any_char().parse("123"), Ok(('1', "23")));
        assert_eq!(any_char().parse("あいう"), Ok(('あ', "いう")));
        assert_eq!(
            any_char().parse(""),
            Err(ParseError {
                position: 0,
                expected: vec!["any char".to_string()],
                found: None,
            })
        );
    }
}

pub fn char(c: char) -> impl Parser<Item = char> {
    pred(move |x| x == c).label(format!("char '{}'", c))
}
#[cfg(test)]
mod test_char {
    use super::*;

    #[test]
    fn test_char() {
        assert_eq!(char('a').parse("abc"), Ok(('a', "bc")));
        assert_eq!(char('あ').parse("あいう"), Ok(('あ', "いう")));
        assert_eq!(
            char('a').parse("bca"),
            Err(ParseError {
                position: 0,
                expected: vec!["char 'a'".to_string()],
                found: Some("b".to_string())
            })
        );
        assert_eq!(
            char('a').parse(""),
            Err(ParseError {
                position: 0,
                expected: vec!["char 'a'".to_string()],
                found: None,
            })
        );
    }
}

pub fn space() -> impl Parser<Item = char> {
    pred(|c: char| c.is_whitespace()).label("whitespace".to_string())
}
#[cfg(test)]
mod test_space {
    use super::*;

    #[test]
    fn test_space() {
        assert_eq!(space().parse(" abc"), Ok((' ', "abc")));
        assert_eq!(space().parse("\tabc"), Ok(('\t', "abc")));
        assert_eq!(space().parse("\nabc"), Ok(('\n', "abc")));
        assert_eq!(
            space().parse("abc"),
            Err(ParseError {
                position: 0,
                expected: vec!["whitespace".to_string()],
                found: Some("a".to_string())
            })
        );
    }
}

pub fn spaces() -> impl Parser<Item = Vec<char>> {
    pred(|c: char| c.is_whitespace()).many0()
}
#[cfg(test)]
mod test_spaces {
    use super::*;

    #[test]
    fn test_spaces() {
        assert_eq!(spaces().parse("  abc"), Ok((vec![' ', ' '], "abc")));
        assert_eq!(spaces().parse("\tabc"), Ok((vec!['\t'], "abc")));
        assert_eq!(spaces().parse("\nabc"), Ok((vec!['\n'], "abc")));
        assert_eq!(
            spaces().parse(" \t\nabc"),
            Ok((vec![' ', '\t', '\n'], "abc"))
        );
        assert_eq!(spaces().parse("abc"), Ok((vec![], "abc")));
    }
}

pub fn keyword(s: &str) -> impl Parser<Item = &str> {
    Keyword(s)
}
fn _keyword<'a>(s: &'a str) -> impl FnOnce(&str) -> ParseResult<&'a str> {
    move |input| {
        if input.starts_with(s) {
            Ok((s, &input[s.len()..]))
        } else {
            Err(ParseError {
                position: 0,
                expected: vec![s.to_string()],
                found: Some(input.to_string()),
            })
        }
    }
}
#[derive(Debug, Clone)]
pub struct Keyword<'a>(&'a str);
impl<'a> Parser for Keyword<'a> {
    type Item = &'a str;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _keyword(self.0)(input)
    }
}
#[cfg(test)]
mod test_keyword {
    use super::*;

    #[test]
    fn test_keyword() {
        assert_eq!(keyword("abc").parse("abcdef"), Ok(("abc", "def")));
        assert_eq!(
            keyword("abc").parse("abdef"),
            Err(ParseError {
                position: 0,
                expected: vec!["abc".to_string()],
                found: Some("abdef".to_string())
            })
        );
        assert_eq!(
            keyword("あいう").parse("あいうえお"),
            Ok(("あいう", "えお"))
        );
    }
}

pub fn string() -> impl Parser<Item = String> {
    Str
}
fn _string() -> impl FnOnce(&str) -> ParseResult<String> {
    move |input| {
        if input.starts_with('"') {
            let mut chars = input.chars();
            let mut len = 0;
            chars.next();
            loop {
                match chars.next() {
                    Some('"') => {
                        len += 1; // +1 means the closing '"'
                        return Ok((input[1..len].to_string(), &input[len + 1..]));
                    }
                    Some(c) => len += c.len_utf8(),
                    None => {
                        return Err(ParseError {
                            position: len + 1,
                            expected: vec![r#"missing closing '"'"#.to_string()],
                            found: Some(input[..len + 1].to_string()),
                        })
                    }
                }
            }
        } else {
            Err(ParseError {
                position: 0,
                expected: vec!["any string".to_string()],
                found: Some(input.to_string()),
            })
        }
    }
}
#[derive(Debug, Clone)]
pub struct Str;
impl Parser for Str {
    type Item = String;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _string()(input)
    }
}
#[cfg(test)]
mod test_string {
    use super::*;

    #[test]
    fn test_string() {
        assert_eq!(
            string().parse(""),
            Err(ParseError {
                position: 0,
                expected: vec!["any string".to_string()],
                found: Some("".to_string()),
            })
        );
        assert_eq!(
            string().parse("abcdef"),
            Err(ParseError {
                position: 0,
                expected: vec!["any string".to_string()],
                found: Some("abcdef".to_string()),
            })
        );
        assert_eq!(
            string().parse("\"abc"),
            Err(ParseError {
                position: 4,
                expected: vec![r#"missing closing '"'"#.to_string()],
                found: Some("\"abc".to_string()),
            })
        );
        assert_eq!(
            string().parse("\""),
            Err(ParseError {
                position: 1,
                expected: vec![r#"missing closing '"'"#.to_string()],
                found: Some("\"".to_string()),
            })
        );
        assert_eq!(
            string().parse("123"),
            Err(ParseError {
                position: 0,
                expected: vec!["any string".to_string()],
                found: Some("123".to_string()),
            })
        );
        assert_eq!(string().parse("\"\"xxx"), Ok(("".to_string(), "xxx")));
        assert_eq!(string().parse("\"abc\"def"), Ok(("abc".to_string(), "def")));
        assert_eq!(
            string().parse("\"Hello, world!\" he said."),
            Ok(("Hello, world!".to_string(), " he said."))
        );
        assert_eq!(
            string().parse("\"\\escaped\\by\\backslash\" and more"),
            Ok(("\\escaped\\by\\backslash".to_string(), " and more"))
        );
        assert_eq!(
            string().parse("\"あいう\"えお"),
            Ok(("あいう".to_string(), "えお"))
        );
    }
}

pub fn constant<T>(c: T) -> impl Parser<Item = T>
where
    T: Clone,
{
    Const(c)
}
fn _constant<T>(c: T) -> impl FnOnce(&str) -> ParseResult<T> {
    move |input| Ok((c, input))
}
#[derive(Debug, Clone)]
pub struct Const<T>(T);
impl<T: Clone> Parser for Const<T> {
    type Item = T;

    fn parse(self, input: &str) -> ParseResult<Self::Item> {
        _constant(self.0)(input)
    }
}
#[cfg(test)]
mod test_constant {
    use super::*;

    #[test]
    fn test_constant() {
        assert_eq!(constant(123).parse("abc"), Ok((123, "abc")));
        assert_eq!(constant("abc").parse("def"), Ok(("abc", "def")));
        assert_eq!(constant("あい").parse("うえお"), Ok(("あい", "うえお")));
        assert_eq!(constant(true).parse("abc"), Ok((true, "abc")));
    }
}
