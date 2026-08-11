---
title: "Triple-Checked Second-Order Kerr Teukolsky Solver Specification"
subtitle: "Corrected equations, independent derivation, Kokkos architecture, and validation protocol"
author: "Technical handoff bundle"
date: "August 11, 2026"
geometry: margin=0.78in
fontsize: 10pt
linestretch: 1.04
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
toc: true
toc-depth: 3
---

## Triple-checked derivation and implementation audit of the second-order Kerr Teukolsky system

**Version:** 2.0.0  
**Date:** 2026-08-11  
**Legacy code audited:** `JLRipley314/teuk-fortran-2020`, commit `0a12faef3461f04ef71c8d42517eaaba512e34aa`  
**Primary papers:** Loutrel et al., arXiv:2008.11770; Ripley et al., arXiv:2010.00162

## Executive conclusion

The continuum formalism used by `teuk-fortran-2020` is internally consistent after fixing one source coefficient. The first-order Teukolsky reduction, the background Newman-Penrose fields, the rescaled GHP operators, the seven outgoing-radiation-gauge metric-reconstruction transport equations, and all other audited terms in the compact quadratic source agree with the papers and with independent symbolic reconstructions.

The legacy second-order path nevertheless must not be used unchanged for a quantitative extremal or near-extremal instability calculation. The audit finds two critical defects:

1. In the compact quadratic source, the connection part of
   
   \[
   \frac12(\eth+\bar\pi+2\tau)h_{l\bar m}
   \]
   
   is implemented with coefficient 1 instead of \(1/2\). The correct rescaled term is
   
   \[
   \frac12R\,\eth_2 C+
   \frac12R^2(\bar\pi_0+2\tau_0)C.
   \]

2. The driven second-order evolution is globally second-order accurate in time. Its midpoint source values are endpoint averages, so the RK4 source quadrature collapses exactly to the trapezoidal rule.

The audit also identifies high-severity engineering defects: reuse of a stale endpoint RHS after filtering, unsafe general multimode bookkeeping, inconsistent dimensional handling of the delayed second-order start time, and the absence of an independent dealiased angular-product oracle.

The companion script `verify_second_order_teukolsky.py` executes 94 algebraic, structural, and manufactured-convergence checks. In the recorded environment all 94 pass with SymPy 1.14.0. This is strong reproducible evidence against sign, factor, radial-falloff, mode-selection, and stage-order transcription errors. It is not a formal proof of the full continuum formalism, and a Mathematica kernel was not available for an independent execution of the repository notebook.

---

## 1. Conventions and scope

We follow the conventions of the two primary papers. The background is Kerr, written in the horizon-penetrating, hyperboloidally compactified coordinates

\[
(T,R,\theta,\phi),\qquad r=\frac{L^2}{R}.
\]

Future null infinity is \(R=0\), and the outer horizon is

\[
R_H=\frac{L^2}{r_+},\qquad
r_+=M+\sqrt{M^2-a^2}.
\]

Every field is Fourier decomposed as

\[
X(T,R,\theta,\phi)=\sum_m X_m(T,R,\theta)e^{im\phi}.
\]

For complex conjugate tetrad components, the correct mode operation is

\[
X_m^\sharp\equiv \overline{X_{-m}}.
\]

This definition must not be replaced by \(\overline{X_m}\) unless a separate reality relation has been proved for that field. The quadratic source must sum all ordered pairs

\[
(m_1,m_2)\longrightarrow m_t=m_1+m_2.
\]

The chosen background tetrad is a rotated Kinnersley tetrad regular at future null infinity, with \(\gamma=0\). The first-order metric is reconstructed in outgoing radiation gauge, with the first-order tetrad conditions used in Appendix B of arXiv:2010.00162. In this gauge and tetrad, \(\nu^{(1)}=\mu^{(1)}=d_4^{(1)}=0\) in the source specialization used below.

The object \(\Psi_4^{(2)}\) is convention fixed but is not, at a generic finite-radius point, invariant under arbitrary first-order gauge and tetrad transformations. At future null infinity the paper's asymptotically flat gauge and tetrad permit a direct radiation interpretation. At the horizon, physical claims require either a geometrically prescribed regular tetrad or observer-based quantities in addition to raw code-tetrad \(\Psi_4^{(2)}\).

---

## 2. Triple-check strategy

The second-order equation was reconstructed through three algebraically independent routes.

### Route I: general type-D second-order NP/GHP equation

Starting from Eqs. (B14) and (B15) of arXiv:2008.11770, the second-order Bianchi identity is expanded before fixing outgoing radiation gauge. The second-order Ricci-identity combination is eliminated with Eq. (B15), completing the background Teukolsky operator and leaving the general source Eq. (B17).

This route checks the global signs, the relative coefficients \(-1,+1,-3,+3\), and the placement of the first-order metric/tetrad corrections without using the specialized source in arXiv:2010.00162.

### Route II: outgoing-radiation-gauge and tetrad specialization

The general source is specialized term by term using Eqs. (B29)-(B36) of arXiv:2010.00162. The calculation groups all contributions into two inner sources and the two background outer operators,

\[
\mathcal S=(\Delta+4\mu+\bar\mu)s_d
+(\eth'+4\pi-\bar\tau)s_t.
\]

This route independently reproduces Eqs. (14), (15a), and (15b) from the Appendix B intermediate expressions.

### Route III: direct elimination of \(\rho^{(1)}\)

The disputed coefficient can be isolated without trusting the final Eq. (15a). Before eliminating \(\rho^{(1)}\), Eq. (B31) contains one full copy of

\[
(\eth+\bar\pi+2\tau)h_{l\bar m}.
\]

Equation (B19) gives the needed first-order spin coefficient. In the convention entering B31, the relevant contribution is

\[
\rho^{(1)}\supset
-\frac12(\eth+\bar\pi+2\tau)h_{l\bar m}.
\]

Therefore substitution gives

\[
1-\frac12=\frac12
\]

multiplying the entire operator combination. The factor does not act only on \(\eth h_{l\bar m}\); it also multiplies the connection term \((\bar\pi+2\tau)h_{l\bar m}\).

The three routes agree.

---

## 3. Route I: general second-order Teukolsky equation

Define the background operators used in the companion formalism paper by

\[
d_4^{(0)}=\Delta+4\mu+\bar\mu,
\]

\[
d_3^{(0)}=\bar\delta+3\alpha+\bar\beta+4\pi-\bar\tau.
\]

The expanded second-order Bianchi identity, Eq. (B14) of arXiv:2008.11770, contains

\[
\begin{aligned}
0={}&
\left[d_4^{(0)}(D+4\epsilon-\rho)^{(0)}
-d_3^{(0)}(\delta+4\beta-\tau)^{(0)}\right]\Psi_4^{(2)}\\
&+\left[d_4^{(0)}(D+4\epsilon-\rho)^{(1)}
-d_3^{(0)}(\delta+4\beta-\tau)^{(1)}\right]\Psi_4^{(1)}\\
&+\left[-d_4^{(0)}(\bar\delta+2\alpha+4\pi)^{(1)}
+d_3^{(0)}(\Delta+2\gamma+4\mu)^{(1)}\right]\Psi_3^{(1)}\\
&+3\left[d_4^{(0)}\lambda^{(1)}-d_3^{(0)}\nu^{(1)}\right]\Psi_2^{(1)}\\
&+3\left[d_4^{(0)}\lambda^{(2)}-d_3^{(0)}\nu^{(2)}\right]\Psi_2^{(0)}.
\end{aligned}
\]

For vacuum perturbations, Eq. (B15) replaces the second-order spin-coefficient combination by

\[
\begin{aligned}
\left[d_4^{(0)}\lambda^{(2)}-d_3^{(0)}\nu^{(2)}\right]\Psi_2^{(0)}
={}&-\Psi_2^{(0)}\Psi_4^{(2)}\\
&+\Psi_2^{(0)}\left[-(d_4^{(1)}-3\mu^{(1)})\lambda^{(1)}
+(d_3^{(1)}-3\pi^{(1)})\nu^{(1)}\right].
\end{aligned}
\]

The term \(-3\Psi_2^{(0)}\Psi_4^{(2)}\) completes the background spin \(-2\) Teukolsky operator. The resulting vacuum equation is

\[
\boxed{
\mathcal T\Psi_4^{(2)}=S_4^{(2)},
}
\]

with

\[
\begin{aligned}
S_4^{(2)}={}&-
\left[d_4^{(0)}(D+4\epsilon-\rho)^{(1)}
-d_3^{(0)}(\delta+4\beta-\tau)^{(1)}\right]\Psi_4^{(1)}\\
&+\left[d_4^{(0)}(\bar\delta+2\alpha+4\pi)^{(1)}
-d_3^{(0)}(\Delta+2\gamma+4\mu)^{(1)}\right]\Psi_3^{(1)}\\
&-3\left[d_4^{(0)}\lambda^{(1)}-d_3^{(0)}\nu^{(1)}\right]\Psi_2^{(1)}\\
&+3\Psi_2^{(0)}\left[(d_4^{(1)}-3\mu^{(1)})\lambda^{(1)}
-(d_3^{(1)}-3\pi^{(1)})\nu^{(1)}\right].
\end{aligned}
\]

The executable audit treats every operator application as an independent algebraic symbol, substitutes Eq. (B15) into Eq. (B14), and verifies this sign ledger exactly. No coordinate or gauge specialization is used in Route I.

---

## 4. Route II: specialized continuum source

Under the outgoing-radiation-gauge and tetrad conditions of arXiv:2010.00162, the last line of the general source vanishes and \(\nu^{(1)}=0\). The surviving source can be written

\[
\boxed{
\mathcal S=(\Delta+4\mu+\bar\mu)s_d
+(\eth'+4\pi-\bar\tau)s_t.
}
\]

The corrected inner source multiplying the \(d_4\) operator is

\[
\boxed{
\begin{aligned}
s_d={}&
\frac12h_{ll}(\Delta+\mu)\Psi_4^{(1)}\\
&+\Psi_4^{(1)}\left[
\frac12(\eth+\bar\pi+2\tau)h_{l\bar m}
+(\Delta-\mu+\bar\mu)h_{ll}
-\frac12(\eth'-5\pi-4\bar\tau)h_{lm}
\right]\\
&-\frac12\Psi_3^{(1)}\left[
(\eth+\bar\pi+\tau)h_{\bar m\bar m}
+(\Delta-2\mu+\bar\mu)h_{l\bar m}
\right]\\
&-\left(h_{l\bar m}\Delta
-\frac12h_{\bar m\bar m}\eth
-4\pi^{(1)}\right)\Psi_3^{(1)}
-3\lambda^{(1)}\Psi_2^{(1)}.
\end{aligned}
}
\]

The inner source multiplying the \(d_3\) operator is

\[
\boxed{
\begin{aligned}
s_t={}&
-h_{lm}(\Delta+\mu+2\bar\mu)\Psi_4^{(1)}
+\frac12h_{mm}\eth'\Psi_4^{(1)}\\
&+\Psi_4^{(1)}\left[
\bar\pi^{(1)}-\Delta h_{lm}
+\left(\eth'-\frac12\pi-\frac12\bar\tau\right)h_{mm}
\right].
\end{aligned}
}
\]

These are Eqs. (14)-(15) of the implementation paper. The paper itself contains a textual typo immediately before Eq. (13), referring to an equation for \(\Psi_2^{(2)}\); the displayed equation and the rest of the work clearly evolve \(\Psi_4^{(2)}\). This does not affect the code equations.

### 4.1 Reconstruction of the \(s_t\) coefficients

Equation (B30) is particularly useful as an independent sign check. Starting from ordinary NP derivatives and then collecting them into GHP derivatives gives

\[
-h_{lm}(\Delta+\mu+2\bar\mu)\Psi_4^{(1)}
+\frac12h_{mm}\eth'\Psi_4^{(1)}
\]

and

\[
\Psi_4^{(1)}\left[
\bar\pi^{(1)}-\Delta h_{lm}
+\left(\eth'-\frac12\pi-\frac12\bar\tau\right)h_{mm}
\right].
\]

The two \(-1/2\) connection coefficients therefore follow before any radial compactification.

### 4.2 Reconstruction of the \(\Psi_3^{(1)}\) and \(\lambda^{(1)}\Psi_2^{(1)}\) terms

Equation (B34) supplies

\[
(\Delta+4\mu+\bar\mu)
\left[
\left(-h_{l\bar m}\Delta
+\frac12h_{\bar m\bar m}\eth
+4\pi^{(1)}\right)\Psi_3^{(1)}
-\frac12\Psi_3^{(1)}
\left((\eth+\bar\pi+\tau)h_{\bar m\bar m}
+(\Delta-2\mu+\bar\mu)h_{l\bar m}\right)
\right].
\]

Equation (B35) supplies

\[
-3(\Delta+4\mu+\bar\mu)
\left(\lambda^{(1)}\Psi_2^{(1)}\right).
\]

Equation (B36) vanishes for the adopted tetrad. These terms reproduce the last two lines of \(s_d\).

---

## 5. Route III: independent proof of the disputed factor

This route is the strongest local check because it derives the factor from two earlier Appendix equations instead of assuming Eq. (15a).

Before eliminating \(\rho^{(1)}\), the first source family contains

\[
\Psi_4^{(1)}
(\eth+\bar\pi+2\tau)h_{l\bar m}
+\Psi_4^{(1)}\rho^{(1)}
\]

with the sign dictated by Eq. (B31). Equation (B19), after converting the conjugated equation to the convention entering B31, contains

\[
\rho^{(1)}=
\frac12\mu h_{ll}
+\frac12(\eth'-\pi)h_{lm}
-\frac12(\eth+\bar\pi+2\tau)h_{l\bar m}.
\]

Substitution gives

\[
\begin{aligned}
&(\eth+\bar\pi+2\tau)h_{l\bar m}
-\frac12(\eth+\bar\pi+2\tau)h_{l\bar m}\\
&\qquad=\frac12(\eth+\bar\pi+2\tau)h_{l\bar m}.
\end{aligned}
\]

Thus the coefficient \(1/2\) applies to both

\[
\eth h_{l\bar m}
\]

and

\[
(\bar\pi+2\tau)h_{l\bar m}.
\]

No algebraic rearrangement permits the derivative piece to carry \(1/2\) while the connection piece carries 1.

---

## 6. Radial compactification and regularized fields

The code evolves regularized fields whose radial powers are chosen to be finite and generally nonzero from the horizon through future null infinity:

\[
\Psi_4^{(1)}=RF,
\qquad
\Psi_3^{(1)}=R^2G,
\qquad
\Psi_2^{(1)}=R^3H,
\]

\[
\lambda^{(1)}=R\Lambda,
\qquad
\pi^{(1)}=R^2\Pi,
\]

\[
h_{\bar m\bar m}=RB,
\qquad
h_{l\bar m}=R^2C,
\qquad
\mu h_{ll}=R^3U.
\]

The last definition means

\[
h_{ll}=\frac{R^2U}{\mu_0},
\qquad \mu=R\mu_0.
\]

The nonzero background quantities needed here are

\[
\mu_0=\frac{1}{-L^2+iaR\cos\theta},
\]

\[
\tau_0=\frac{ia\sin\theta}
{\sqrt2(L^2-iaR\cos\theta)^2},
\]

\[
\pi_0=-\frac{ia\sin\theta}
{\sqrt2(L^4+a^2R^2\cos^2\theta)},
\]

\[
\psi_{20}=-\frac{M}{(L^2-iaR\cos\theta)^3}.
\]

An exact background identity used to avoid a singular-looking derivative of \(h_{ll}\) is

\[
\boxed{\Delta\mu=-\mu^2.}
\]

It implies

\[
(\Delta-\mu+\bar\mu)h_{ll}
=\frac{1}{\mu}\Delta(\mu h_{ll})+\bar\mu h_{ll}.
\]

### 6.1 Rescaled operators

For a field with physical radial factor \(R^n\), define

\[
\Delta_n X=R^{-n}\Delta(R^nX),
\]

\[
\eth_nX=R^{-(n+1)}\eth(R^nX),
\qquad
\eth'_nX=R^{-(n+1)}\eth'(R^nX).
\]

Because the ingoing tetrad vector is

\[
n^\mu\partial_\mu=
\left(2+\frac{4MR}{L^2}\right)\partial_T
+\frac{R^2}{L^2}\partial_R,
\]

we have

\[
\boxed{
\Delta_nX=
\left(2+\frac{4MR}{L^2}\right)\partial_TX
+\frac{R}{L^2}(R\partial_RX+nX).
}
\]

Let \(s\) and \(b\) be spin and boost weights and define \(p=s+b\), \(q=-s+b\). With the unit-sphere raising and lowering actions

\[
\mathcal R_s\,{}_sY_{\ell m}
=\sqrt{(\ell-s)(\ell+s+1)}\,{}_{s+1}Y_{\ell m},
\]

\[
\mathcal L_s\,{}_sY_{\ell m}
=-\sqrt{(\ell+s)(\ell-s+1)}\,{}_{s-1}Y_{\ell m},
\]

the compact-coordinate GHP operators are

\[
\boxed{
\eth_nX=
\frac{-ia\sin\theta\,\partial_TX+\mathcal R_sX}
{\sqrt2(L^2-iaR\cos\theta)}
-\frac{ipaR\sin\theta}
{\sqrt2(L^2-iaR\cos\theta)^2}X,
}
\]

\[
\boxed{
\eth'_nX=
\frac{ia\sin\theta\,\partial_TX+\mathcal L_sX}
{\sqrt2(L^2+iaR\cos\theta)}
+\frac{iqaR\sin\theta}
{\sqrt2(L^2+iaR\cos\theta)^2}X.
}
\]

The audit verifies that these connection factors are algebraically identical to the expressions in `mod_ghp.f90`.

---

## 7. Correct compact ordered-pair source

Define regular inner sources

\[
D=\frac{s_d}{R^3},
\qquad
T=\frac{s_t}{R^3}.
\]

For each ordered pair \((m_1,m_2)\), with target \(m_t=m_1+m_2\), define

\[
\begin{aligned}
D_{12}={}&
\frac12U_1\left(\frac{\Delta_1F_2}{\mu_0}+RF_2\right)\\
&+F_1\Bigg[
\frac12R\,\eth_2C_2
+\frac12R^2(\bar\pi_0+2\tau_0)C_2
+\frac{\Delta_3U_2}{\mu_0}
+R\,U_2^\sharp\\
&\qquad
-\frac12R\,\eth'_2C_2^\sharp
+\frac12R^2(5\pi_0+4\bar\tau_0)C_2^\sharp
\Bigg]\\
&-\frac12G_1\Bigg[
R\,\eth_1B_2
+R^2(\bar\pi_0+\tau_0)B_2
+R\,\Delta_2C_2
+R^2(-2\mu_0+\bar\mu_0)C_2
\Bigg]\\
&-R\,C_1\Delta_2G_2
+\frac12R\,B_1\eth_2G_2
+4R\,\Pi_1G_2
-3R\,\Lambda_1H_2.
\end{aligned}
\]

The second inner source is

\[
\begin{aligned}
T_{12}={}&
-C_1^\sharp
\left[\Delta_1F_2+R(\mu_0+2\bar\mu_0)F_2\right]
+\frac12B_1^\sharp\eth'_1F_2\\
&+F_1\left[
\Pi_2^\sharp
-\Delta_2C_2^\sharp
+\eth'_1B_2^\sharp
-\frac12R(\pi_0+\bar\tau_0)B_2^\sharp
\right].
\end{aligned}
\]

Sum all ordered pairs:

\[
D_{m_t}=\sum_{m_1+m_2=m_t}D_{12},
\qquad
T_{m_t}=\sum_{m_1+m_2=m_t}T_{12}.
\]

The outer source is

\[
\boxed{
\frac{\mathcal S_{m_t}}{R^3}
=\Delta_3D_{m_t}
+R(4\mu_0+\bar\mu_0)D_{m_t}
+R\eth'_3T_{m_t}
+R^2(4\pi_0-\bar\tau_0)T_{m_t}.
}
\]

The forcing used in the coordinate first-order Teukolsky system is

\[
\boxed{
\mathcal F^{(2)}_{m_t}
=2(L^4+a^2R^2\cos^2\theta)
\frac{\mathcal S_{m_t}}{R^3}.
}
\]

### 7.1 Exact legacy mismatch

The legacy source contains

\[
\frac12R\,\eth_2C_2
+R^2(\bar\pi_0+2\tau_0)C_2
\]

inside the \(F_1\) bracket. The excess relative to the corrected equation is

\[
\boxed{
\delta D_{\rm legacy}
=\frac12R^2F_1(\bar\pi_0+2\tau_0)C_2.
}
\]

This error vanishes at \(a=0\), which helps explain why a Schwarzschild-only regression cannot detect it. It is generically nonzero for Kerr. At the equator,

\[
\bar\pi_0+2\tau_0
=\frac{3ia}{\sqrt2L^4}.
\]

The correction is therefore directly relevant near the horizon of a rapidly rotating black hole.

### 7.2 Term-by-term rescaling check

Every term in \(s_d\) and \(s_t\) has physical radial power at least \(R^3\). Dividing by \(R^3\) produces exactly the explicit factors in \(D_{12}\) and \(T_{12}\). The machine-readable ledger `SOURCE_TERM_LEDGER.csv` records, for every monomial:

- its continuum origin;
- spin and boost weights;
- physical radial power;
- compact expression;
- code status;
- whether it is part of the confirmed defect.

The symbolic audit verifies the falloff of each source family independently rather than checking only the final total.

---

## 8. Metric reconstruction equations

The seven outgoing-radiation-gauge transport equations are required because the second-order source depends on the full first-order metric and selected first-order Weyl/spin-coefficient fields.

The physical equations are

\[
-(\Delta+4\mu)\Psi_3^{(1)}+(\eth-\tau)\Psi_4^{(1)}=0,
\]

\[
-(\Delta+\mu+\bar\mu)\lambda^{(1)}-\Psi_4^{(1)}=0,
\]

\[
-(\Delta+3\mu)\Psi_2^{(1)}+(\eth-2\tau)\Psi_3^{(1)}=0,
\]

\[
-(\Delta-\mu+\bar\mu)h_{\bar m\bar m}-2\lambda^{(1)}=0,
\]

\[
-\Delta\pi^{(1)}-\Psi_3^{(1)}
-(\bar\pi+\tau)\lambda^{(1)}
+\frac12\mu(\bar\pi+\tau)h_{\bar m\bar m}=0,
\]

\[
-(\Delta+\bar\mu)h_{l\bar m}
-2\pi^{(1)}-\tau h_{\bar m\bar m}=0,
\]

\[
\begin{aligned}
0={}&-(\Delta+\bar\mu)(\mu h_{ll})
-\mu(\eth+\bar\pi+2\tau)h_{l\bar m}
-2\Psi_2^{(1)}\\
&-\pi(\eth'-\pi)h_{mm}
+(\mu\eth'-3\mu\pi+2\bar\mu\pi-2\mu\bar\tau)h_{lm}\\
&-2(\eth+\bar\pi)\pi^{(1)}-2\pi\bar\pi^{(1)}.
\end{aligned}
\]

Their regularized forms are

\[
\boxed{\Delta_2G=-4R\mu_0G+\eth_1F-R\tau_0F,}
\]

\[
\boxed{\Delta_1\Lambda=-R(\mu_0+\bar\mu_0)\Lambda-F,}
\]

\[
\boxed{\Delta_3H=-3R\mu_0H+\eth_2G-2R\tau_0G,}
\]

\[
\boxed{\Delta_1B=R(\mu_0-\bar\mu_0)B-2\Lambda,}
\]

\[
\boxed{
\Delta_2\Pi=-G-R(\bar\pi_0+\tau_0)\Lambda
+\frac12R^2\mu_0(\bar\pi_0+\tau_0)B,
}
\]

\[
\boxed{\Delta_2C=-R\bar\mu_0C-2\Pi-R\tau_0B,}
\]

\[
\boxed{
\begin{aligned}
\Delta_3U={}&-R\bar\mu_0U
-R\mu_0\eth_2C
-R^2\mu_0(\bar\pi_0+2\tau_0)C
-2\eth_2\Pi-2R\bar\pi_0\Pi-2H\\
&-2R\pi_0\Pi^\sharp
-R\pi_0\eth'_1B^\sharp
+R^2\pi_0^2B^\sharp
+R\mu_0\eth'_2C^\sharp\\
&+R^2(-3\mu_0\pi_0+2\bar\mu_0\pi_0-2\mu_0\bar\tau_0)C^\sharp.
\end{aligned}
}
\]

For a generic transport equation \(\Delta_nX=\mathcal R_X\), solve for the coordinate time derivative using

\[
\boxed{
\partial_TX=
\frac{\mathcal R_X-(R^2/L^2)\partial_RX-(nR/L^2)X}
{2+4MR/L^2}.
}
\]

The audit maps every term in these seven equations to `mod_metric_recon.f90` and finds no coefficient mismatch.

---

## 9. Linear and driven Teukolsky first-order system

For spin weight \(s\) and Fourier mode \(m\), let

\[
C_T=8M\left(2M-\frac{a^2R}{L^2}\right)
\left(1+\frac{2MR}{L^2}\right)-a^2\sin^2\theta,
\]

\[
K=L^2-\frac{(8M^2-a^2)R^2}{L^2}
+\frac{4a^2MR^3}{L^4},
\]

\[
H_R=\frac{R^2}{L^4}(L^4-2L^2MR+a^2R^2),
\]

and

\[
\begin{aligned}
G_m={}&2iam\left(1+\frac{4MR}{L^2}\right)\\
&+2\left[
2M\left(-s+(2+s)\frac{2MR}{L^2}
-\frac{3a^2R^2}{L^4}\right)
-\frac{a^2R}{L^2}+isa\cos\theta
\right].
\end{aligned}
\]

Define

\[
P=C_T\partial_T\psi-2K\partial_R\psi+G_m\psi,
\qquad
Q=\partial_R\psi.
\]

Then

\[
\boxed{
\partial_T\psi=\frac{P+2KQ-G_m\psi}{C_T}.
}
\]

The \(P\) equation is

\[
\begin{aligned}
\partial_TP={}&H_R\partial_RQ\\
&+\left[
2R\left(1+s-(3+s)\frac{MR}{L^2}
+\frac{2a^2R^2}{L^4}\right)
-\frac{2iamR^2}{L^2}
\right]Q\\
&+\left[
-2R\left((1+s)\frac{M}{L^2}
-\frac{a^2R}{L^4}\right)
-\frac{2iamR}{L^2}
\right]\psi
+{}_{s}\!\Delta_{\Omega}\,\psi+\mathcal F.
\end{aligned}
\]

Use \(\mathcal F=0\) for the first-order field and \(\mathcal F=\mathcal F^{(2)}\) for the driven second-order field.

Rather than copying the long expanded legacy \(Q\) coefficients, generate the reduction equation from the defining relation:

\[
\boxed{
\partial_TQ=\partial_R
\left(\frac{P+2KQ-G_m\psi}{C_T}\right)
-\gamma_Q(Q-\partial_R\psi).
}
\]

Alternatively impose \(Q=\partial_R\psi\) at every RK stage. Both strategies should be implemented and compared. The symbolic audit confirms that the legacy expanded \(Q\) coefficients are consistent with the derivative form, but the derivative form is safer and easier to verify.

The radial principal coefficient \(H_R\) vanishes at \(R=0\) and at the outer horizon. This supports a characteristic treatment without arbitrary physical boundary data, but the exact semi-discrete SBP-SAT closure must still be derived and tested.

---

## 10. Time integration: why the legacy driven scheme is order two

The legacy implementation first advances the complete first-order state to \(T_{n+1}\), computes the endpoint source, and supplies the second-order RK stages with

\[
S_1=S_n,
\qquad
S_2=S_3=\frac{S_n+S_{n+1}}2,
\qquad
S_4=S_{n+1}.
\]

The source contribution to classical RK4 is therefore

\[
\begin{aligned}
\Delta U_S
&=\frac{\Delta T}{6}(S_1+2S_2+2S_3+S_4)\\
&=\frac{\Delta T}{2}(S_n+S_{n+1}),
\end{aligned}
\]

which is exactly the trapezoidal rule. Expanding about \(T_n\),

\[
S(T_n+\Delta T)=S_0+S_1\Delta T
+\frac12S_2\Delta T^2+\cdots,
\]

the local forcing defect starts at

\[
\frac{S''(T_n)}{12}\Delta T^3,
\]

so the global driven solution is order two.

The five-point backward difference used to estimate some source time derivatives is itself fourth order. That fact does not restore fourth-order accuracy to endpoint-averaged source quadrature.

The audit includes a coupled manufactured ODE,

\[
x'=x,
\qquad
y'=-0.3y+x^2,
\]

for which correctly stage-coupled RK4 converges at order 4.00 while the legacy source staging converges at order 2.00.

### 10.1 Correct stage-local construction

All systems must share the same RK stages and abscissae. At every stage:

1. construct the first-order Teukolsky stage state;
2. evaluate its RHS;
3. construct and evaluate all seven reconstruction fields at that stage;
4. build the inner source primitives \(D\) and \(T\);
5. evaluate the time derivatives entering \(\Delta_3D\) and \(\eth'_3T\) from that same stage;
6. form \(\mathcal S\) and \(\mathcal F^{(2)}\);
7. evaluate the second-order Teukolsky RHS;
8. accumulate all variables with the same RK coefficients.

Because the first-order and reconstruction systems are stationary and linear,

\[
\dot U=L(U),
\qquad
\ddot U=L(\dot U).
\]

For every bilinear source primitive \(B(U,V)\), its exact stage tangent is

\[
\boxed{
\partial_TB(U,V)=B(\dot U,V)+B(U,\dot V).
}
\]

A small forward-mode type such as

```cpp
struct Jet1Complex {
  Kokkos::complex<double> value;
  Kokkos::complex<double> dt;
};
```

or an explicitly generated Jacobian-vector product can propagate these derivatives without history-based finite differences. Classical coupled RK4 should be the correctness path. Low-storage or fused variants should be adopted only after full fourth-order driven convergence is established.

---

## 11. Other legacy implementation risks

### 11.1 Stale endpoint derivative after filtering

The legacy code computes \(k_5=F(U_{n+1})\), filters or projects \(U_{n+1}\), and then reuses \(k_5\) as the first derivative of the next step. In general,

\[
F(\mathcal P U)\ne F(U).
\]

For the second-order source this can mix filtered endpoint fields with pre-filter time derivatives. The new code should place compatible dissipation in the method-of-lines RHS. If any projection is applied after a step, all dependent RHS and derivative caches must be invalidated and recomputed.

### 11.2 General multimode bookkeeping

The legacy top-level evolution assumes a particular relationship between `lin_m` and `lin_pos_m`, while setup code constructs some collections through unordered sets. This is unsafe once more than one \(|m|\) is evolved. The new code must use a sorted signed-mode registry, an explicit mode-to-index map, explicit conjugate lookup, and a precomputed ordered-pair table. The \(m=0\) mode must be advanced exactly once.

### 11.3 Delayed source activation and units

The wait-time logic mixes a dimensionless \(T_{\rm mw}/M\) estimate with coordinate time in a way that depends on the chosen numerical value of \(M\). Use \(M=1\) internally or consistently dimensional strong types. Treat source activation as an RK event; align it to a time step or split the crossing step.

### 11.4 Angular aliasing

The old code uses angular overcollocation and filtering but has no independent exact-product test. For source products with

\[
s_3=s_1+s_2,
\qquad
m_3=m_1+m_2,
\]

the modal oracle is the generalized spin-weighted Gaunt coefficient

\[
\begin{aligned}
&\int {}_{s_1}Y_{\ell_1m_1}
{}_{s_2}Y_{\ell_2m_2}
\overline{{}_{s_3}Y_{\ell_3m_3}}\,d\Omega\\
&=(-1)^{m_3+s_3}
\sqrt{\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\-s_1&-s_2&s_3\end{pmatrix}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\m_1&m_2&-m_3\end{pmatrix}.
\end{aligned}
\]

A padded collocation product is appropriate for production, but it must agree with this oracle over the supported bandlimit.

---

## 12. Numerical design for the near-extremal instability

The relevant linear effect is the Aretakis horizon instability. For extremal Kerr, nonaxisymmetric modes can dominate and exhibit enhanced transverse-derivative growth controlled by the mode's conformal weight. More generally, extremal Kerr perturbations admit near-horizon late-time self-similar scaling with computable critical exponents. The field itself need not form a finite-time jump; increasingly high transverse derivatives grow along the horizon.

For that reason, the baseline radial method should be a high-order linear SBP finite-difference discretization with SAT boundary treatment and compatible dissipation. Spectral angular treatment remains appropriate because the expected loss of regularity is transverse to the horizon, not angular. WENO or MP5 should not be the default: nonlinear dissipation may suppress the effect being measured.

A sensible progression is:

1. uniform-\(R\) D4-2 SBP reference implementation;
2. D6-3 SBP after verification;
3. smooth horizon-clustering map or two radial SBP blocks;
4. only then, optional nonlinear radial stencils if convergence demonstrates an actual non-smooth field.

Exact extremality and near extremality are different numerical campaigns. For near-extremal Kerr organize late-time results using the surface gravity

\[
\kappa=\frac{r_+-r_-}{2(r_+^2+a^2)}
\]

and the scaled time \(\kappa v\).

The second-order exponent should not be assumed to be twice the first-order exponent. The source contains fields, transverse derivatives, reconstruction variables, connection terms, cancellations, angular convolutions, and a response through the second-order Green function. A derivative-growing factor times a decaying field may dominate over a naive \((\Psi_4^{(1)})^2\) estimate, and a near-horizon or zero-damped response can add secular enhancement.

Required diagnostics therefore include:

- \(\Psi_4^{(1)}\) and \(\Psi_4^{(2)}\) in raw and horizon-corotating/demodulated forms;
- transverse derivatives through at least fourth order;
- every named quadratic source family separately;
- spin-weighted modal projections;
- first-order reduction and reconstruction residuals;
- source amplitude-squared scaling;
- perturbative validity ratios comparing the physical first- and second-order contributions;
- a regular horizon tetrad or infalling-observer diagnostic;
- exact-extremal and near-extremal convergence sequences.

---

## 13. Verification evidence

The executable audit performs the following independent checks:

- B14+B15 to B16+B17 source algebra;
- the four general-source signs and coefficients;
- B29-B36 to Eqs. (14)-(15);
- direct B31+B19 proof of the disputed \(1/2\);
- every radial falloff in \(s_d\) and \(s_t\);
- the regularized outer source;
- \(\Delta\mu=-\mu^2\);
- background Kerr NP fields;
- GHP connection coefficients;
- coordinate forcing normalization;
- first-order Teukolsky coefficient identities;
- the derivative form of the \(Q\) equation;
- horizon and scri limits;
- all seven reconstruction equations;
- ordered mode-pair enumeration;
- spin-weighted Gaunt selection rules and normalization tests;
- exact reduction of the legacy source staging to the trapezoidal rule;
- a fourth-order versus second-order manufactured convergence comparison;
- the bilinear source JVP identity.

Recorded result:

```text
Completed 94 checks with SymPy 1.14.0: 94 passed, 0 failed.
```

The audit intentionally distinguishes three categories:

- **confirmed continuum/code mismatch:** the missing \(1/2\) on the Kerr connection term;
- **confirmed numerical-algorithm defect:** global second-order driven time accuracy;
- **verification or engineering risk:** stale caches, mode ordering, source activation units, and angular aliasing.

It does not claim that no unexamined bug can exist. In particular, the full Fortran program was not rebuilt and rerun across all published configurations in this audit environment, and the Mathematica notebook was not independently executed. The package instead provides transparent symbolic scripts, equation ledgers, and tests that the new implementation can retain permanently.

---

## 14. Files that constitute the source of truth for the new implementation

Codex should read these files in order:

1. `README.md`
2. `TRIPLE_CHECKED_SECOND_ORDER_TEUKOLSKY_DERIVATION.md`
3. `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`
4. `equation_spec_v2.yaml`
5. `SOURCE_TERM_LEDGER.csv`
6. `verify_second_order_teukolsky.py`
7. `verify_second_order_teukolsky_output.txt`
8. `IMPLEMENTATION_AND_VALIDATION_PLAN.md`
9. `CODEX_PROMPT_KOKKOS_SECOND_ORDER_TEUKOLSKY_V2.md`

The papers and legacy repositories remain indispensable references for context and regression data, but they are not the new code's executable source of truth. The corrected equation specification and tests in this bundle are.

---

## References

1. N. Loutrel, J. L. Ripley, E. Giorgi, and F. Pretorius, *Second Order Perturbations of Kerr Black Holes: Reconstruction of the Metric*, arXiv:2008.11770.
2. J. L. Ripley, N. Loutrel, E. Giorgi, and F. Pretorius, *Numerical computation of second order vacuum perturbations of Kerr black holes*, arXiv:2010.00162.
3. M. Casals, S. E. Gralla, and P. Zimmerman, *Horizon Instability of Extremal Kerr Black Holes: Nonaxisymmetric Modes and Enhanced Growth Rate*, arXiv:1606.08505.
4. S. E. Gralla and P. Zimmerman, *Critical Exponents of Extremal Kerr Perturbations*, arXiv:1711.00855.
5. S. A. Teukolsky, *Perturbations of a rotating black hole. I. Fundamental equations for gravitational, electromagnetic, and neutrino-field perturbations*, Astrophys. J. 185, 635 (1973).

\newpage

## Corrected equations: implementation reference for a second-order Kerr Teukolsky solver

**Specification version:** 2.0.0  
**Date:** 2026-08-11  
**Legacy snapshot audited:** `JLRipley314/teuk-fortran-2020`, commit `0a12faef3461f04ef71c8d42517eaaba512e34aa`

This document is the compact implementation reference. The longer derivation document explains how each equation was obtained and independently checked. The executable audit is `verify_second_order_teukolsky.py`; the same equations are also encoded in `equation_spec_v2.yaml` and `SOURCE_TERM_LEDGER.csv`.

## 1. Conventions that must be fixed before coding

Use the horizon-penetrating, hyperboloidally compactified coordinates of Ripley et al.:

\[
r=\frac{L^2}{R},\qquad R=0\ \text{at future null infinity},\qquad R_H=\frac{L^2}{r_+}.
\]

Decompose every field as

\[
X(T,R,\theta,\phi)=\sum_m X_m(T,R,\theta)e^{im\phi}.
\]

For mode conjugation define

\[
X_m^{\sharp}\equiv \overline{X_{-m}}.
\]

Do not replace this rule with `conj(X_m)` unless a real-field relation has been proved for the specific NP/tetrad component. Store all signed modes explicitly and evaluate every ordered source pair

\[
(m_1,m_2)\longrightarrow m_t=m_1+m_2.
\]

Internally set \(M=1\) unless the entire code uses a dimensionally consistent strong-unit system. Convert only at input/output.

## 2. Spin, boost, and radial rescalings

Use the following rescaled first-order fields:

\[
\begin{array}{c|c|c|c|c}
\text{physical field}&\text{rescaled field}&s&b&\text{definition}\\ \hline
\Psi_4^{(1)}&F&-2&-2&\Psi_4^{(1)}=RF\\
\Psi_3^{(1)}&G&-1&-1&\Psi_3^{(1)}=R^2G\\
\Psi_2^{(1)}&H&0&0&\Psi_2^{(1)}=R^3H\\
\lambda^{(1)}&\Lambda&-2&-1&\lambda^{(1)}=R\Lambda\\
\pi^{(1)}&\Pi&-1&0&\pi^{(1)}=R^2\Pi\\
h_{\bar m\bar m}&B&-2&0&h_{\bar m\bar m}=RB\\
h_{l\bar m}&C&-1&1&h_{l\bar m}=R^2C\\
\mu h_{ll}&U&0&1&\mu h_{ll}=R^3U
\end{array}
\]

The last line is deliberate: the evolved variable is \(\mu h_{ll}\), not \(h_{ll}\) itself. Since \(\mu=R\mu_0\), this implies

\[
h_{ll}=\frac{R^2U}{\mu_0}.
\]

The needed background quantities are

\[
\mu=R\mu_0,
\qquad
\mu_0=\frac{1}{-L^2+iaR\cos\theta},
\]

\[
\tau=R^2\tau_0,
\qquad
\tau_0=\frac{ia\sin\theta}{\sqrt2\,(L^2-iaR\cos\theta)^2},
\]

\[
\pi=R^2\pi_0,
\qquad
\pi_0=-\frac{ia\sin\theta}{\sqrt2\,[L^4+a^2R^2\cos^2\theta]},
\]

\[
\Psi_2^{(0)}=R^3\psi_{20},
\qquad
\psi_{20}=-\frac{M}{(L^2-iaR\cos\theta)^3}.
\]

A useful exact identity is

\[
\boxed{\Delta\mu=-\mu^2.}
\]

It gives

\[
(\Delta-\mu+\bar\mu)h_{ll}
=\frac{1}{\mu}\Delta(\mu h_{ll})+\bar\mu h_{ll}.
\]

## 3. Rescaled GHP operators

For a field \(X\) with radial falloff \(R^n\), define

\[
\Delta_n X\equiv R^{-n}\Delta(R^nX),
\]

\[
\eth_nX\equiv R^{-(n+1)}\eth(R^nX),
\qquad
\eth'_nX\equiv R^{-(n+1)}\eth'(R^nX).
\]

In the coordinates used here,

\[
\boxed{
\Delta_nX=
\left(2+\frac{4MR}{L^2}\right)\partial_TX
+\frac{R}{L^2}\left(R\partial_RX+nX\right).
}
\]

Let \(s\) and \(b\) be spin and boost weights, and define \(p=s+b\), \(q=-s+b\). If \(\mathcal R_s\) and \(\mathcal L_s\) are the unit-sphere spin-raising and lowering matrices,

\[
\mathcal R_s\,{}_sY_{\ell m}
=\sqrt{(\ell-s)(\ell+s+1)}\,{}_{s+1}Y_{\ell m},
\]

\[
\mathcal L_s\,{}_sY_{\ell m}
=-\sqrt{(\ell+s)(\ell-s+1)}\,{}_{s-1}Y_{\ell m},
\]

then

\[
\boxed{
\eth_nX=
\frac{-ia\sin\theta\,\partial_TX+\mathcal R_sX}
{\sqrt2\,(L^2-iaR\cos\theta)}
-\frac{ipaR\sin\theta}
{\sqrt2\,(L^2-iaR\cos\theta)^2}X,
}
\]

\[
\boxed{
\eth'_nX=
\frac{ia\sin\theta\,\partial_TX+\mathcal L_sX}
{\sqrt2\,(L^2+iaR\cos\theta)}
+\frac{iqaR\sin\theta}
{\sqrt2\,(L^2+iaR\cos\theta)^2}X.
}
\]

The label \(n\) is radial-falloff bookkeeping; the displayed angular formulas have no explicit \(n\) because the angular tetrad vector contains no radial derivative.

## 4. Correct continuum second-order source

The second-order field obeys

\[
\boxed{\mathcal T\Psi_4^{(2)}=\mathcal S,}
\]

with the same background spin-\(-2\) Teukolsky operator as at first order and

\[
\boxed{
\mathcal S=(\Delta+4\mu+\bar\mu)s_d
+(\eth'+4\pi-\bar\tau)s_t.
}
\]

The correct inner sources in outgoing radiation gauge are

\[
\begin{aligned}
s_d={}&
\frac12h_{ll}(\Delta+\mu)\Psi_4^{(1)}\\
&+\Psi_4^{(1)}\left[
\frac12(\eth+\bar\pi+2\tau)h_{l\bar m}
+(\Delta-\mu+\bar\mu)h_{ll}
-\frac12(\eth'-5\pi-4\bar\tau)h_{lm}
\right]\\
&-\frac12\Psi_3^{(1)}\left[
(\eth+\bar\pi+\tau)h_{\bar m\bar m}
+(\Delta-2\mu+\bar\mu)h_{l\bar m}
\right]\\
&-\left(h_{l\bar m}\Delta
-\frac12h_{\bar m\bar m}\eth
-4\pi^{(1)}\right)\Psi_3^{(1)}
-3\lambda^{(1)}\Psi_2^{(1)},
\end{aligned}
\]

\[
\begin{aligned}
s_t={}&
-h_{lm}(\Delta+\mu+2\bar\mu)\Psi_4^{(1)}
+\frac12h_{mm}\eth'\Psi_4^{(1)}\\
&+\Psi_4^{(1)}\left[
\bar\pi^{(1)}-\Delta h_{lm}
+\left(\eth'-\frac12\pi-\frac12\bar\tau\right)h_{mm}
\right].
\end{aligned}
\]

### Critical correction to the legacy code

The factor \(1/2\) multiplies the **entire** operator combination

\[
\boxed{
\frac12(\eth+\bar\pi+2\tau)h_{l\bar m}.
}
\]

After radial rescaling, the correct two pieces are

\[
\boxed{
\frac12R\,\eth_2C
+\frac12R^2(\bar\pi_0+2\tau_0)C.
}
\]

The legacy source uses coefficient \(1/2\) on the first piece and coefficient \(1\) on the second. Its excess is

\[
\boxed{
\delta D_{\rm legacy}
=\frac12R^2F_1(\bar\pi_0+2\tau_0)C_2.
}
\]

This is nonzero for rotating Kerr. At \(\cos\theta=0\),

\[
\bar\pi_0+2\tau_0=\frac{3ia}{\sqrt2L^4}.
\]

## 5. Correct compactified ordered-pair source

Define

\[
D\equiv \frac{s_d}{R^3},\qquad T\equiv\frac{s_t}{R^3}.
\]

For one ordered pair \((m_1,m_2)\), with target \(m_t=m_1+m_2\), the corrected expression is

\[
\begin{aligned}
D_{12}={}&
\frac12U_1\left(\frac{\Delta_1F_2}{\mu_0}+RF_2\right)\\
&+F_1\Bigg[
\frac12R\,\eth_2C_2
+\frac12R^2(\bar\pi_0+2\tau_0)C_2
+\frac{\Delta_3U_2}{\mu_0}
+R\,U_2^{\sharp}\\
&\hspace{3.8em}
-\frac12R\,\eth'_2C_2^{\sharp}
+\frac12R^2(5\pi_0+4\bar\tau_0)C_2^{\sharp}
\Bigg]\\
&-\frac12G_1\Bigg[
R\,\eth_1B_2
+R^2(\bar\pi_0+\tau_0)B_2
+R\,\Delta_2C_2
+R^2(-2\mu_0+\bar\mu_0)C_2
\Bigg]\\
&-R\,C_1\Delta_2G_2
+\frac12R\,B_1\eth_2G_2
+4R\,\Pi_1G_2
-3R\,\Lambda_1H_2.
\end{aligned}
\]

The transverse inner source is

\[
\begin{aligned}
T_{12}={}&
-C_1^{\sharp}\left[
\Delta_1F_2+R(\mu_0+2\bar\mu_0)F_2
\right]
+\frac12B_1^{\sharp}\eth'_1F_2\\
&+F_1\left[
\Pi_2^{\sharp}
-\Delta_2C_2^{\sharp}
+\eth'_1B_2^{\sharp}
-\frac12R(\pi_0+\bar\tau_0)B_2^{\sharp}
\right].
\end{aligned}
\]

Sum all ordered pairs:

\[
D_{m_t}=\sum_{m_1+m_2=m_t}D_{12},
\qquad
T_{m_t}=\sum_{m_1+m_2=m_t}T_{12}.
\]

Then

\[
\boxed{
\frac{\mathcal S_{m_t}}{R^3}
=\Delta_3D_{m_t}
+R(4\mu_0+\bar\mu_0)D_{m_t}
+R\eth'_3T_{m_t}
+R^2(4\pi_0-\bar\tau_0)T_{m_t}.
}
\]

The forcing appearing in the coordinate \((P,Q,\psi)\) equation is

\[
\boxed{
\mathcal F^{(2)}_{m_t}
=2\left(L^4+a^2R^2\cos^2\theta\right)
\frac{\mathcal S_{m_t}}{R^3}.
}
\]

## 6. First-order metric reconstruction

The physical outgoing-radiation-gauge transport equations are

\[
-(\Delta+4\mu)\Psi_3^{(1)}+(\eth-\tau)\Psi_4^{(1)}=0,
\]

\[
-(\Delta+\mu+\bar\mu)\lambda^{(1)}-\Psi_4^{(1)}=0,
\]

\[
-(\Delta+3\mu)\Psi_2^{(1)}+(\eth-2\tau)\Psi_3^{(1)}=0,
\]

\[
-(\Delta-\mu+\bar\mu)h_{\bar m\bar m}-2\lambda^{(1)}=0,
\]

\[
-\Delta\pi^{(1)}-\Psi_3^{(1)}
-(\bar\pi+\tau)\lambda^{(1)}
+\frac12\mu(\bar\pi+\tau)h_{\bar m\bar m}=0,
\]

\[
-(\Delta+\bar\mu)h_{l\bar m}
-2\pi^{(1)}-\tau h_{\bar m\bar m}=0,
\]

\[
\begin{aligned}
0={}&-(\Delta+\bar\mu)(\mu h_{ll})
-\mu(\eth+\bar\pi+2\tau)h_{l\bar m}
-2\Psi_2^{(1)}\\
&-\pi(\eth'-\pi)h_{mm}
+(\mu\eth'-3\mu\pi+2\bar\mu\pi-2\mu\bar\tau)h_{lm}\\
&-2(\eth+\bar\pi)\pi^{(1)}-2\pi\bar\pi^{(1)}.
\end{aligned}
\]

Their rescaled form is

\[
\boxed{\Delta_2G=-4R\mu_0G+\eth_1F-R\tau_0F,}
\]

\[
\boxed{\Delta_1\Lambda=-R(\mu_0+\bar\mu_0)\Lambda-F,}
\]

\[
\boxed{\Delta_3H=-3R\mu_0H+\eth_2G-2R\tau_0G,}
\]

\[
\boxed{\Delta_1B=R(\mu_0-\bar\mu_0)B-2\Lambda,}
\]

\[
\boxed{
\Delta_2\Pi=-G-R(\bar\pi_0+\tau_0)\Lambda
+\frac12R^2\mu_0(\bar\pi_0+\tau_0)B,
}
\]

\[
\boxed{\Delta_2C=-R\bar\mu_0C-2\Pi-R\tau_0B,}
\]

\[
\boxed{
\begin{aligned}
\Delta_3U={}&-R\bar\mu_0U
-R\mu_0\eth_2C
-R^2\mu_0(\bar\pi_0+2\tau_0)C
-2\eth_2\Pi
-2R\bar\pi_0\Pi
-2H\\
&-2R\pi_0\Pi^{\sharp}
-R\pi_0\eth'_1B^{\sharp}
+R^2\pi_0^2B^{\sharp}
+R\mu_0\eth'_2C^{\sharp}\\
&+R^2(-3\mu_0\pi_0+2\bar\mu_0\pi_0-2\mu_0\bar\tau_0)C^{\sharp}.
\end{aligned}
}
\]

For each equation solve \(\Delta_nX=R_X\) as

\[
\partial_TX=
\frac{R_X-(R^2/L^2)\partial_RX-(nR/L^2)X}
{2+4MR/L^2}.
\]

## 7. Linear and driven Teukolsky first-order reduction

For spin weight \(s\) and Fourier mode \(m\), define

\[
C_T=8M\left(2M-\frac{a^2R}{L^2}\right)
\left(1+\frac{2MR}{L^2}\right)-a^2\sin^2\theta,
\]

\[
K=L^2-\frac{(8M^2-a^2)R^2}{L^2}
+\frac{4a^2MR^3}{L^4},
\]

\[
H_R=\frac{R^2}{L^4}(L^4-2L^2MR+a^2R^2),
\]

\[
\begin{aligned}
G_m={}&2iam\left(1+\frac{4MR}{L^2}\right)\\
&+2\left[
2M\left(-s+(2+s)\frac{2MR}{L^2}-\frac{3a^2R^2}{L^4}\right)
-\frac{a^2R}{L^2}+isa\cos\theta
\right].
\end{aligned}
\]

Define

\[
P=C_T\partial_T\psi-2K\partial_R\psi+G_m\psi,
\qquad Q=\partial_R\psi.
\]

Then

\[
\boxed{
\partial_T\psi=\frac{P+2KQ-G_m\psi}{C_T}.
}
\]

The \(P\) equation is

\[
\begin{aligned}
\partial_TP={}&H_R\partial_RQ\\
&+\left[
2R\left(1+s-(3+s)\frac{MR}{L^2}+\frac{2a^2R^2}{L^4}\right)
-\frac{2iamR^2}{L^2}
\right]Q\\
&+\left[
-2R\left((1+s)\frac{M}{L^2}-\frac{a^2R}{L^4}\right)
-\frac{2iamR}{L^2}
\right]\psi
+{}_{s}\!\Delta_{\Omega}\,\psi+\mathcal F.
\end{aligned}
\]

Use \(\mathcal F=0\) at first order and the corrected \(\mathcal F^{(2)}\) above at second order.

Do not copy the expanded legacy \(Q\) coefficients. Generate

\[
\boxed{
\partial_TQ=
\partial_R\left(\frac{P+2KQ-G_m\psi}{C_T}\right)
-\gamma_Q(Q-\partial_R\psi),
}
\]

or impose \(Q=\partial_R\psi\) at every RK stage. Implement both strategies and cross-check them.

## 8. Nonlinear angular product oracle

For a product with \(s_3=s_1+s_2\) and \(m_3=m_1+m_2\), use the generalized spin-weighted Gaunt coefficient

\[
\begin{aligned}
&\int {}_{s_1}Y_{\ell_1m_1}
{}_{s_2}Y_{\ell_2m_2}
\overline{{}_{s_3}Y_{\ell_3m_3}}\,d\Omega\\
&=(-1)^{m_3+s_3}
\sqrt{\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\-s_1&-s_2&s_3\end{pmatrix}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\m_1&m_2&-m_3\end{pmatrix}.
\end{aligned}
\]

The production padded-collocation product may be faster, but it must agree with this modal oracle over the supported bandlimit. Overcollocation without this test is not sufficient evidence of dealiasing.

## 9. Required fourth-order stage coupling

The legacy driven scheme uses

\[
S_1=S_n,\qquad
S_2=S_3=\frac{S_n+S_{n+1}}2,\qquad
S_4=S_{n+1}.
\]

Its RK4 source contribution is exactly

\[
\frac{\Delta T}{6}(S_1+2S_2+2S_3+S_4)
=\frac{\Delta T}{2}(S_n+S_{n+1}),
\]

so the driven solution is only globally second order for a time-dependent source.

The new solver must evaluate, at every common RK stage:

1. the first-order Teukolsky RHS;
2. the seven reconstruction RHSs;
3. all inner-source primitives \(D\) and \(T\);
4. the time derivatives needed inside \(\Delta\), \(\eth\), and \(\eth'\);
5. the outer source \(\mathcal S\);
6. the second-order Teukolsky RHS.

Because the background system is stationary and linear, at a stage state \(U\),

\[
\dot U=L(U),\qquad \ddot U=L(\dot U).
\]

For every bilinear source primitive \(B(U,V)\), evaluate its tangent by

\[
\boxed{
\partial_TB(U,V)=B(\dot U,V)+B(U,\dot V).
}
\]

A `Jet1<T>{value, dt}` representation or a manually generated Jacobian-vector product is appropriate. Start with classical coupled RK4 as the correctness oracle. Optimize to a low-storage scheme only after fourth-order convergence is demonstrated for the full driven system.

## 10. Numerical method requirements for the instability study

Use finite differences only in \(R\), retaining the spin-weighted spectral angular representation and explicit \(m\) decomposition.

Recommended progression:

- a verified diagonal-norm SBP operator with SAT boundary treatment as the reference discretization;
- fourth-order interior/second-order boundary closure first, followed by sixth-order interior/third-order boundary closure;
- compatible KO/SBP dissipation included in the method-of-lines RHS;
- no post-step filter followed by reuse of a pre-filter RHS;
- optional two-block or mapped near-horizon refinement before nonlinear WENO/MP5;
- nonlinear radial stencils only after convergence evidence shows an actual loss of smoothness rather than merely large transverse derivatives.

For exact extremality and near extremality, separately output

\[
\Psi_4^{(1)},\quad \Psi_4^{(2)},\quad
\partial_R^n\Psi_4^{(1,2)}\quad (n\ge 4),
\]

all source families and angular modes, horizon-corotating/demodulated fields, reconstruction and reduction residuals, and the near-extremal scaling variable \(\kappa v\).

Raw horizon \(\Psi_4^{(2)}\) in the code tetrad is a convention-fixed diagnostic, not generally a first-order gauge/tetrad invariant. Include a regular horizon tetrad or infalling-observer diagnostic before making a physical instability claim.

## 11. Minimal regression tests that must exist before production runs

1. The executable symbolic audit passes.
2. A focused test fails when the corrected connection coefficient \(1/2\) is changed to \(1\).
3. A forced manufactured problem shows fourth-order time convergence; the legacy staging reproduces order two.
4. The source scales as amplitude squared.
5. Every source term has the correct spin/boost weight and radial falloff.
6. Ordered \(m\)-pair enumeration is deterministic and includes both orderings.
7. Padded angular products agree with the Gaunt oracle.
8. The Schwarzschild limit removes all disputed Kerr connection terms.
9. Constraint and reconstruction residuals converge at the designed order.
10. CPU reference and Kokkos CPU/GPU kernels agree within a justified tolerance.
11. Restarted and uninterrupted evolutions agree.
12. Exact-extremal and near-extremal test problems are run separately.

\newpage

## Implementation and validation plan for the Kokkos second-order Teukolsky solver

**Version:** 2.0.0  
**Date:** 2026-08-11

This document turns the corrected equations into an executable development sequence. It is intentionally more prescriptive than a normal design note because the target calculation is a long-time, near-extremal, weakly nonlinear instability study where a small source or integration error can masquerade as secular growth.

## 1. Scientific objective

Construct a performance-portable time-domain solver for

1. the first-order spin \(-2\) Teukolsky equation on Kerr;
2. outgoing-radiation-gauge metric reconstruction by seven nested null-transport equations;
3. the corrected quadratic source for \(\Psi_4^{(2)}\);
4. the driven second-order spin \(-2\) Teukolsky equation;
5. exact-extremal and near-extremal horizon diagnostics.

The baseline numerical representation is

\[
\text{finite differences/SBP in }R
\quad+\quad
\text{spin-weighted spectral treatment in }\theta
\quad+\quad
e^{im\phi}\text{ modes}.
\]

The implementation language is C++ with Kokkos. The same physics kernels must compile for CPU and available CUDA, HIP, and SYCL execution spaces without backend-specific equation code.

## 2. Source-of-truth hierarchy

Use the following hierarchy when references disagree:

1. executable checks in `verify_second_order_teukolsky.py`;
2. `equation_spec_v2.yaml` and `SOURCE_TERM_LEDGER.csv`;
3. `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`;
4. the derivation in this package;
5. the two papers;
6. legacy Fortran/Julia code, used only for regression and convention comparison.

Do not copy expanded coefficients from the legacy code without deriving or generating them from the compact equations.

## 3. Repository architecture

Recommended initial structure:

```text
CMakeLists.txt
cmake/
configs/
docs/
  CONVENTIONS.md
  EQUATIONS.md
  ARCHITECTURE.md
  VALIDATION_MATRIX.md
  LEGACY_AUDIT.md
  PERFORMANCE.md
  STATE.md
  DECISIONS.md
include/teuk/
  core/
  geometry/
  angular/
  radial/
  evolution/
  reconstruction/
  source/
  diagnostics/
  io/
src/
tools/symbolic/
tools/reference/
tests/unit/
tests/integration/
tests/convergence/
benchmarks/
examples/
```

The symbolic tools should generate checked C++ headers or machine-readable coefficient files. Generated files must contain provenance hashes and must be checked in only together with their generators and tests.

## 4. Typed data model

### 4.1 Mode registry

Create an immutable registry containing:

- sorted unique signed \(m\) values;
- map \(m\to i_m\);
- conjugate lookup \(m\to -m\);
- nonnegative representative list used only where mathematically valid;
- target-mode list;
- deterministic ordered source pairs \((m_1,m_2,m_t)\);
- validation that every required input mode exists.

Never infer pairings from storage order. Handle \(m=0\) exactly once.

### 4.2 Field metadata

Every field type should carry compile-time or immutable runtime metadata:

- spin weight \(s\);
- boost weight \(b\);
- radial falloff \(n\);
- physical meaning;
- whether \(X^\sharp_m=\overline{X_{-m}}\) is required;
- angular bandlimit;
- current state generation/version for cache validation.

### 4.3 Device storage

Start with a benchmarked structure-of-arrays design. A reasonable baseline is

```cpp
using complex_type = Kokkos::complex<double>;
using view4_type = Kokkos::View<complex_type****, Kokkos::LayoutRight,
                                memory_space>;
// dimensions: mode, variable, radial, theta
```

with \(\theta\) contiguous. Also benchmark one View per field. Use accessors so the numerics do not depend strongly on one layout, but do not add abstraction that prevents device inlining.

Keep stationary background coefficients and angular matrices on the device. Do not transfer full fields to the host per stage.

## 5. Numerical components

### 5.1 Background and coordinates

Implement exact coordinate maps, horizon location, surface gravity, tetrad coefficients, spin coefficients, and rescaled background fields. Tests must cover:

- Schwarzschild limit;
- regularity at \(R=0\) and \(R=R_H\);
- \(\Delta\mu=-\mu^2\);
- agreement with symbolic formulas at randomized points;
- dimensionless rescaling under changes of \(M\) and \(L\).

Prefer \(M=1\) internally. If arbitrary \(M\) is supported, add explicit dimensional tests.

### 5.2 Spin-weighted angular representation

Implement modal and nodal representations at fixed \(m\). Required operations:

- synthesis and analysis on a Gauss-Legendre grid;
- \(\mathcal R_s\) and \(\mathcal L_s\);
- spin-weighted Laplacian;
- filtered/padded transforms for products;
- exact generalized Gaunt oracle for tests;
- spin/boost/falloff metadata checks on every operator.

Test orthonormality, round trips, eigenvalues, raising/lowering factors, adjoints, selection rules, and pole regularity.

### 5.3 Radial SBP operators

Correctness path:

1. diagonal-norm D4-2 first derivative;
2. compatible dissipation and SAT interface;
3. D6-3 after the vertical slice is verified;
4. optional smooth horizon-clustering map;
5. optional two-block SBP-SAT grid.

Tests must prove

\[
HD+D^TH=B
\]

to roundoff, polynomial exactness through the formal order, nonpositivity of dissipation in the SBP norm, and designed convergence for mapped and unmapped grids.

The baseline system is first order in radial derivatives using \((P,Q,\psi)\), so no second-derivative SBP operator is required for the production formulation. The Julia second-order-in-space code can remain a separate regression reference.

### 5.4 Boundary treatment

Derive characteristic speeds at future null infinity and the outer horizon from the implemented first-order system. Because the relevant radial principal coefficient vanishes at both ends, do not impose arbitrary Dirichlet conditions. Use SBP closures and SAT penalties only for genuinely incoming or reduction-constraint modes.

Before production, include a semi-discrete energy or normal-mode study for a frozen-coefficient representative system and long-time numerical tests with outgoing pulses.

### 5.5 Dissipation and filtering

Place compatible KO/SBP dissipation inside the RHS. Do not apply a post-step filter and reuse an old RHS. If a projection is unavoidable, increment the state generation and invalidate every derivative/source cache.

Make dissipation strength configurable and include it in convergence and instability sensitivity studies. The physical growth rate must be stable under a range that controls grid noise without dominating the solution.

## 6. Coupled equations and stage semantics

### 6.1 First-order Teukolsky state

Evolve \((P^{(1)},Q^{(1)},F)\) for each signed \(m\). Support both:

- free reduction evolution with damping;
- stage-wise constrained reconstruction \(Q=\partial_RF\).

Cross-check the two approaches on smooth solutions.

### 6.2 Reconstruction state

Evolve, in dependency order,

\[
G,\ \Lambda,\ H,\ B,\ \Pi,\ C,\ U.
\]

At every RK stage, later reconstruction equations use the same-stage values and same-stage time derivatives of earlier variables. This dependency chain must be explicit in code, not hidden in mutable global field objects.

### 6.3 Quadratic source state

Represent each named source family separately. Suggested identifiers mirror `SOURCE_TERM_LEDGER.csv`, for example:

```text
D_hll_DeltaF
D_hll_muF
D_F_ethC
D_F_connection_C
D_F_DeltaU
D_F_conjugate_U
...
T_conjugate_C_DeltaF
T_conjugate_B_ethprimeF
...
```

Expose optional per-family output. Sum only after each family has passed spin, boost, falloff, amplitude-squared, and operator-oracle tests.

### 6.4 Stage-local time derivatives

The outer operators require \(\partial_TD\) and \(\partial_TT\). Never reconstruct these with historical backward differences. Use one of these equivalent correctness paths:

1. `Jet1<T>{value, dt}` forward-mode propagation;
2. explicitly generated source JVPs;
3. exact algebraic elimination using the first-order/reconstruction equations.

The first implementation should use `Jet1` or explicit JVPs because it preserves the compact operator structure and is easy to compare with a CPU oracle.

### 6.5 Common-stage RK4

Use classical RK4 first. At stage \(a\):

1. build all first-order stage fields;
2. evaluate the first-order RHS;
3. build and evaluate reconstruction fields in dependency order;
4. evaluate all angular/radial derived quantities;
5. evaluate source values and tangents for every ordered mode pair;
6. form \(\mathcal S_a\) and \(\mathcal F_a^{(2)}\);
7. evaluate the second-order RHS;
8. accumulate all systems with the same \(a\), \(b\), and \(c\) coefficients.

No subsystem may be advanced to \(T_{n+1}\) before another subsystem evaluates its midpoint source.

After full fourth-order convergence is demonstrated, benchmark a low-storage fourth-order method. Retain classical RK4 as an oracle.

## 7. Independent CPU oracle

Implement a transparent CPU reference that prioritizes correspondence with the equations over performance. It should:

- use the compact operator-form source;
- use explicit loops and simple data structures;
- retain named source terms;
- support arbitrary smooth randomized fields;
- evaluate both values and stage tangents;
- serve as a pointwise oracle for Kokkos kernels.

Do not make the optimized Kokkos implementation and oracle share the same expanded generated expression if that would allow a single generator error to affect both. At least one path must preserve the compact equation structure.

## 8. Verification matrix

### 8.1 Symbolic tests

- all 94 bundled checks pass;
- generated C++ formulas match the YAML specification;
- background identities and falloffs;
- all field spin and boost weights;
- source family radial powers;
- corrected \(1/2\) factor;
- deliberate legacy coefficient fails;
- \(Q\)-equation derivative identities;
- horizon and scri principal limits.

### 8.2 Discrete angular tests

- orthonormality and transform round trip;
- raising/lowering factors and signs;
- angular Laplacian eigenvalues;
- \(X_m^\sharp=\overline{X_{-m}}\) implementation;
- ordered mode-pair selection;
- padded products versus Gaunt oracle;
- an intentionally unpadded test that shows aliasing, proving the oracle has power.

### 8.3 Discrete radial tests

- SBP identity and polynomial exactness;
- formal boundary closure order;
- dissipation energy sign;
- SAT stability tests;
- mapped-grid convergence;
- two-block interpolation/penalty conservation if implemented.

### 8.4 Time tests

- autonomous RK4 order four;
- explicitly forced ODE order four;
- coupled source-generating ODE order four;
- legacy endpoint-average method order two;
- stage-local JVP against finite-difference directional derivative;
- event-aligned delayed source activation;
- cache-version assertions.

### 8.5 PDE manufactured solutions

Create manufactured solutions for:

1. homogeneous Teukolsky equation with forcing added analytically;
2. each reconstruction transport equation separately;
3. the full dependency chain;
4. a synthetic bilinear source and driven second-order system;
5. coupled angular modes with a known Gaunt product.

Measure spatial and temporal orders separately by holding the other error negligible.

### 8.6 Physical regressions

- Schwarzschild evolution and source simplifications;
- moderate-spin comparison against both legacy codes while the solution is smooth;
- amplitude scaling \(F^{(1)}\propto A\), \(\mathcal S\propto A^2\), \(F^{(2)}\propto A^2\);
- source decomposition consistency;
- future-null-infinity waveform normalization;
- reconstruction residual convergence;
- restart versus uninterrupted run;
- CPU versus GPU backend parity.

Agreement with legacy second-order output is not an acceptance criterion until the source coefficient and stage integration differences are accounted for.

## 9. Near-extremal and extremal campaign design

### 9.1 Exact extremal sequence

Treat \(a=M\) as a dedicated configuration. Verify coordinate and coefficient regularity analytically and numerically. Use a radial resolution sequence and monitor

\[
\partial_R^nF|_{\mathcal H^+},\qquad
\partial_R^nF^{(2)}|_{\mathcal H^+},
\qquad n=0,1,2,3,4,\ldots
\]

in both raw and corotating/demodulated form. Fit powers only over windows stable under resolution, dissipation, extraction-point, and fitting-window changes.

### 9.2 Near-extremal sequence

Use a sequence such as

\[
a/M=0.9,\ 0.99,\ 0.999,\ 0.9999,\ldots
\]

subject to resolution. Record \(\kappa\), use \(\kappa v\), and test collapse of transient growth curves. The duration and radial width of the near-horizon transient should determine whether mapped or multiblock refinement is needed.

### 9.3 Source-family scaling

For each dominant \((\ell,m)\) and source family, record:

- local horizon power or transient scaling;
- radial localization;
- phase relative to the first-order field;
- contribution to each target \((\ell,m)\);
- response in \(F^{(2)}\).

Do not infer the quadratic response exponent solely from \(|F^{(1)}|^2\).

### 9.4 Perturbative validity

Introduce an explicit bookkeeping amplitude \(\epsilon\) and monitor, for chosen norms or observables,

\[
\frac{\epsilon^2|F^{(2)}|}{\epsilon|F^{(1)}|}
=\epsilon\frac{|F^{(2)}|}{|F^{(1)}|}.
\]

Report the time at which second-order corrections become comparable to first order for each chosen \(\epsilon\). This is essential if the quadratic instability grows faster.

## 10. Kokkos performance plan

### 10.1 Kernel decomposition

Initial kernels:

- background coefficient initialization;
- radial SBP derivative plus pointwise linear RHS, fused where practical;
- angular transform/operator application using `TeamPolicy` with league index \((m,R)\);
- reconstruction dependency kernels;
- ordered-pair source kernels;
- source tangent/JVP kernels;
- norm and diagnostic reductions.

Avoid materializing every derivative array. Fuse a stencil with immediate coefficient use when this does not make the reference path unreadable. Keep the oracle unfused.

### 10.2 Backend portability

Template on execution and memory spaces. Do not use raw CUDA/HIP/SYCL code in the physics layer. Backend-specific tuning may live behind small policy traits and must preserve a common test suite.

### 10.3 Benchmarking

For each important variant record:

- backend/device and compiler;
- \(N_m,N_R,N_\theta,N_{\rm var}\);
- wall time per RK stage and per physical time unit;
- effective grid-point updates/s;
- memory footprint;
- kernel launch count;
- source pair count;
- numerical difference from the oracle.

Benchmark before choosing dense angular strategy, field layout, source fusion, or low-storage integration.

### 10.4 MPI

Single-node correctness and performance come first. Radial slab decomposition is the preferred first MPI route because radial stencils require compact halos while angular mode coupling remains local to a slab. Provide device-aware MPI when available and host staging otherwise. Avoid decomposition by \(m\) until a communication model justifies it.

## 11. Milestones and gates

### M0: provenance and repository bootstrap

- copy the audit bundle into `docs/reference/` or another immutable location;
- record checksums;
- establish CMake, Kokkos, test harness, sanitizers, formatting, and state documents;
- run the Python audit in CI.

**Gate:** clean configure/build/test on a CPU backend.

### M1: background, modes, and angular operators

- coordinates and background fields;
- mode registry;
- spin-weighted transforms and GHP angular actions;
- Gaunt oracle.

**Gate:** all unit tests, randomized point tests, and product selection tests pass.

### M2: radial SBP and homogeneous linear evolution

- D4-2 SBP and compatible dissipation;
- first-order \((P,Q,F)\) system;
- characteristic boundary treatment;
- manufactured and pulse tests.

**Gate:** designed convergence and stable long-time moderate-spin evolution.

### M3: metric reconstruction

- seven stage-local transport equations;
- residuals and manufactured solutions;
- source-independent output.

**Gate:** all reconstruction residuals converge and CPU/Kokkos agree.

### M4: corrected quadratic source oracle

- named source families;
- ordered mode-pair table;
- value and tangent/JVP evaluation;
- padded angular products;
- explicit regression for the factor error.

**Gate:** operator oracle and optimized source agree; amplitude-squared and Gaunt tests pass.

### M5: coupled second-order vertical slice

- common-stage RK4;
- driven \((P^{(2)},Q^{(2)},F^{(2)})\);
- delayed source event support;
- checkpoint/restart.

**Gate:** full coupled manufactured solution converges fourth order in time; legacy staging test shows second order.

### M6: moderate-spin scientific regression

- compare with published/legacy smooth cases after accounting for the corrected source and timing;
- source-family output and waveform normalization;
- backend parity.

**Gate:** discrepancies are explained quantitatively and documented.

### M7: extremal/near-extremal infrastructure

- horizon-clustered map or multiblock option as required;
- high-order transverse derivative diagnostics;
- regular horizon tetrad/observer diagnostics;
- \(\kappa v\) output.

**Gate:** resolution and dissipation studies support stable fitted growth laws.

### M8: performance and scale

- kernel fusion/layout tuning;
- D6-3 option;
- available GPU backends;
- optional MPI radial decomposition.

**Gate:** performance report includes correctness deltas and reproducible commands.

## 12. Acceptance criteria for a research-ready release

A release suitable for the proposed instability study must satisfy all of the following:

1. the bundled symbolic audit passes unmodified;
2. the corrected connection coefficient is locked by a focused regression;
3. the full driven system demonstrates fourth-order temporal convergence;
4. the angular product path agrees with exact Gaunt convolution;
5. the radial operator satisfies its SBP identity and designed convergence;
6. reconstruction and reduction residuals converge;
7. CPU reference and every supported Kokkos backend agree within justified tolerances;
8. source amplitude-squared scaling holds;
9. exact-extremal and near-extremal configurations are not conflated;
10. growth fits are stable under resolution, dissipation, fitting window, and radial map changes;
11. gauge/tetrad status of every reported observable is documented;
12. restart, deterministic mode ordering, and provenance metadata are verified;
13. performance results never replace correctness evidence.

\newpage

## References and provenance

**Bundle version:** 2.0.0  
**Date:** 2026-08-11

## Primary mathematical sources

1. N. Loutrel, J. L. Ripley, E. Giorgi, and F. Pretorius, *Second Order Perturbations of Kerr Black Holes: Reconstruction of the Metric*, arXiv:2008.11770. The general second-order source is Eq. (B17), obtained from Eqs. (B14)-(B15).
2. J. L. Ripley, N. Loutrel, E. Giorgi, and F. Pretorius, *Numerical computation of second order vacuum perturbations of Kerr black holes*, arXiv:2010.00162. The specialized source is Eqs. (14)-(15); the independent derivation used Eqs. (B19), (B29)-(B36), and the coordinate forcing in Eq. (35).
3. M. Casals, S. E. Gralla, and P. Zimmerman, *Horizon Instability of Extremal Kerr Black Holes: Nonaxisymmetric Modes and Enhanced Growth Rate*, arXiv:1606.08505.
4. S. E. Gralla and P. Zimmerman, *Critical Exponents of Extremal Kerr Perturbations*, arXiv:1711.00855.
5. S. A. Teukolsky, *Perturbations of a rotating black hole. I. Fundamental equations for gravitational, electromagnetic, and neutrino-field perturbations*, Astrophys. J. 185, 635 (1973).

## Legacy code snapshot audited

Repository: `JLRipley314/teuk-fortran-2020`  
Commit: `0a12faef3461f04ef71c8d42517eaaba512e34aa`

Files used in the audit, with Git blob identifiers returned by GitHub:

- `src/mod_scd_order_source.f90`  
  Blob: `15bc1206422d0d862a6cacdb5479be8dc771656b`  
  Role: compact quadratic source, history derivative, and mode-pair accumulation.
- `src/mod_teuk.f90`  
  Blob: `a1864e987572708c4511a0ad1fbbdbf608c13560`  
  Role: linear/driven first-order reduction and RK staging.
- `src/mod_metric_recon.f90`  
  Blob: `8ef16c656b339f0b56cd20b4720a3a2a51a128cd`  
  Role: seven metric-reconstruction transport equations.
- `src/mod_ghp.f90`  
  Blob: `c56eb7ce458f0590f0fca438efc8d6c243f4c7b6`  
  Role: rescaled GHP operators.
- `src/mod_bkgrd.f90`  
  Blob: `9036c71d9bfa8d53cd5e91f35f24151cfe778f9c`  
  Role: background Kerr NP fields.
- `src/mod_field.f90`  
  Blob: `65f6deca83253af54be2af8c4729fe965e8294d5`  
  Role: RK state/cache semantics.

The legacy repository is a regression reference only. For the new solver, the source-of-truth order is:

1. `TRIPLE_CHECKED_SECOND_ORDER_TEUKOLSKY_DERIVATION.md`
2. `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`
3. `equation_spec_v2.yaml`
4. `SOURCE_TERM_LEDGER.csv`
5. `verify_second_order_teukolsky.py`

## Independent audit routes

- **Route I:** expand the general type-D second-order Bianchi/Ricci system and recover Eq. (B17) of arXiv:2008.11770.
- **Route II:** impose the outgoing-radiation-gauge and tetrad conditions, then collect Eqs. (B29)-(B36) into Eqs. (14)-(15) of arXiv:2010.00162.
- **Route III:** isolate the disputed coefficient by substituting Eq. (B19) directly into Eq. (B31), without using the final printed Eq. (15a).

All three routes give the same source. Route III proves that the factor `1/2` multiplies both the derivative and connection pieces of `(eth + bar(pi) + 2 tau) h_l_barm`.

## Reproducibility note

The executable audit uses exact SymPy algebra wherever possible, plus a manufactured forced-ODE convergence test. A Mathematica kernel was unavailable, so the Mathematica notebook was inspected for provenance but was not independently executed. This limitation is recorded rather than hidden.
