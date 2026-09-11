import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Lenis from 'lenis';
import { StylePicker } from './components/StylePicker';
import 'lenis/dist/lenis.css';
import './landing.css';
import './landing-themes.css';

gsap.registerPlugin(ScrollTrigger);

const tasks = [
  { number: '01', title: 'Name the piece.', tag: 'ITEM TYPE', text: 'A tee, a dress, or something else? Start with what you see.', image: '84', name: 'T-shirt', color: 'lilac' },
  { number: '02', title: 'Find its season.', tag: 'SEASON', text: 'From warm days to cool layers. Explore its predicted season.', image: '177', name: 'Floral dress', color: 'yellow' },
  { number: '03', title: 'Read the details.', tag: 'GENDER & OCCASION', text: 'See the catalogue categories that help describe a piece.', image: '93', name: 'Leather strap watch', color: 'blue' },
  { number: '04', title: 'Follow the feeling.', tag: 'VISUAL SEARCH', text: 'Crop your photo. Find pieces with a similar shape, colour, or texture.', image: '173', name: 'Brown handbag', color: 'pink' },
];

function Arrow() {
  return <span aria-hidden="true">↗</span>;
}

export default function Landing() {
  const root = useRef<HTMLDivElement>(null);
  const [motion, setMotion] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setMotion(!preference.matches);
    preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    if (!motion) return;
    const lenis = new Lenis({ anchors: true, duration: 1.05, prevent: node => Boolean(node.closest('[data-slot="select-content"]')) });
    const tick = (time: number) => lenis.raf(time * 1000);
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add(tick);
    const ctx = gsap.context(() => {
      gsap.from('.landing-title span', { yPercent: 110, rotate: 7, stagger: .055, duration: 1.1, ease: 'power4.out' });
      gsap.from('.landing-hero-note, .landing-intro', { y: 20, opacity: 0, duration: .8, delay: .4 });
      gsap.from('.landing-look', { y: 70, opacity: 0, stagger: .13, duration: 1, delay: .35, ease: 'power3.out' });
      gsap.to('.landing-look--left .landing-product', { yPercent: -12, rotation: -9, ease: 'none', scrollTrigger: { trigger: '.landing-collage', start: 'top bottom', end: 'bottom top', scrub: 1 } });
      gsap.to('.landing-look--right .landing-product', { yPercent: 12, rotation: 12, ease: 'none', scrollTrigger: { trigger: '.landing-collage', start: 'top bottom', end: 'bottom top', scrub: 1 } });
      gsap.utils.toArray<HTMLElement>('.landing-reveal').forEach(element => {
        gsap.from(element, { y: 42, opacity: 0, duration: .85, ease: 'power3.out', scrollTrigger: { trigger: element, start: 'top 92%', once: true } });
      });
    }, root);
    let refreshFrame = 0;
    const observer = new MutationObserver(() => {
      cancelAnimationFrame(refreshFrame);
      refreshFrame = requestAnimationFrame(() => ScrollTrigger.refresh());
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-style'] });
    return () => { observer.disconnect(); cancelAnimationFrame(refreshFrame); ctx.revert(); gsap.ticker.remove(tick); lenis.destroy(); };
  }, [motion]);

  return <div className="landing" ref={root} data-motion={motion}>
    <a className="landing-skip" href="#landing-main">Skip to content</a>
    <header className="landing-nav">
      <a className="landing-logo" href="/" aria-label="Thread home"><span aria-hidden="true">✳</span> THREAD</a>
      <span className="landing-nav-caption">A NEW WAY TO LOOK AT FASHION</span>
      <nav aria-label="Main navigation"><a href="#how-it-works">How it works</a><StylePicker /><a className="landing-nav-cta" href="/demo">Try the demo <Arrow /></a></nav>
    </header>

    <main id="landing-main" className="landing-main" tabIndex={-1}>
      <section className="landing-hero" aria-labelledby="landing-heading">
        <div className="landing-hero-note"><span>FASHION MEETS MACHINE LEARNING</span><span>BUILT FROM SCRATCH. MADE TO EXPLORE.</span></div>
        <h1 id="landing-heading" className="landing-title" aria-label="Thread.">{'THREAD.'.split('').map((letter, index) => <span key={index} aria-hidden="true">{letter}</span>)}</h1>
        <div className="landing-intro"><p>Good style catches your eye.<br /><span>Now see what a model sees.</span></p><p>One photo. Four ways to explore it.<br />Discover the details. Find a familiar feel.</p><a className="landing-text-link" href="/demo">Let’s take a look <Arrow /></a></div>
        <div className="landing-collage">
          <a href="/demo" className="landing-look landing-look--left" aria-label="Try the demo with a fashion photo"><span className="landing-look-label">01 / THE EVERYDAY</span><span className="landing-back-type" aria-hidden="true">A<br />GOOD<br />FIT.</span><img className="landing-product" src="/samples/84.webp" alt="White graphic T-shirt" width="300" height="300" /><span className="landing-look-bottom">Small details. Big personality. <Arrow /></span></a>
          <div className="landing-look landing-look--photo"><img src="/landing/street-style.jpg" alt="Street style portrait in a blue coat, white boots, and pink shoulder bag" width="1100" height="1650" fetchPriority="high" /><span className="landing-photo-caption">STYLE IS IN THE EYE OF THE BEHOLDER.</span><span className="landing-stamp">LOOK<br />CLOSER <Arrow /></span></div>
          <a href="/demo" className="landing-look landing-look--right" aria-label="Explore visual search in the demo"><span className="landing-look-label">02 / THE FINISHING TOUCH</span><span className="landing-orbit" aria-hidden="true">✳</span><img className="landing-product" src="/samples/173.webp" alt="Brown handbag with leather handles" width="300" height="300" /><span className="landing-look-bottom">Find something that feels like you. <Arrow /></span></a>
        </div>
        <div className="landing-caption"><span>A SMALL EXPERIMENT IN SEEING THINGS DIFFERENTLY.</span><a href="#how-it-works">SCROLL TO EXPLORE ↓</a></div>
      </section>

      <section id="how-it-works" className="landing-discover">
        <div className="landing-section-top landing-reveal"><span>THE PROJECT / 01—04</span><span>ONE IMAGE. A FEW NEW PERSPECTIVES.</span></div>
        <div className="landing-section-heading landing-reveal"><h2>More than<br /><em>meets the eye.</em></h2><p>Thread turns a fashion photo into a starting point. Our models explore what a piece is, when it fits, and what else looks like it.</p></div>
        <div className="landing-task-grid">{tasks.map(task => <article className="landing-task landing-reveal" key={task.number}>
          <a href="/demo" className={`landing-task-image landing-${task.color}`} aria-label={`Try Task ${task.number}: ${task.tag.toLowerCase()}`}><span>{task.number}</span><img src={`/samples/${task.image}.webp`} alt={task.name} width="300" height="300" loading="lazy" /><span className="landing-task-arrow"><Arrow /></span></a>
          <div className="landing-task-tag">TASK {task.number} / {task.tag}</div><h3>{task.title}</h3><p>{task.text}</p>
        </article>)}</div>
      </section>

      <section className="landing-process" aria-labelledby="process-heading">
        <div className="landing-process-heading landing-reveal"><span>LESS GUESSWORK. MORE DISCOVERY.</span><h2 id="process-heading">Your photo.<br />A fresh perspective.</h2><a className="landing-pill" href="/demo">Try it for yourself <Arrow /></a></div>
        <div className="landing-steps">{[
          ['Bring a piece.', 'Upload a photo or start with a sample. A clear view of one item is a good place to begin.'],
          ['Get a closer look.', 'Explore the predicted item type, season, gender category, and occasion.'],
          ['See what connects.', 'Adjust the crop, then search the product gallery for visually similar pieces.'],
        ].map(([title, description], index) => <div className="landing-step landing-reveal" key={title}><span>0{index + 1}</span><div><h3>{title}</h3><p>{description}</p></div><Arrow /></div>)}</div>
      </section>

      <section className="landing-about landing-reveal"><span>CURIOUS BY DESIGN.</span><p>A student project about fashion, machine learning, and the things a model notices. Sometimes insightful. Sometimes unexpected. <em>Always worth a closer look.</em></p><div><span>4 TASKS · 5 MODELS · ONE SHARED DATASET</span><p>Predictions are suggestions, not certainty. Gender refers to product categories. The demo uses a local model server.</p><a href="https://github.com/TrnLin/MLA2" target="_blank" rel="noreferrer">Explore the project <Arrow /></a></div></section>

      <section className="landing-outro"><div className="landing-reveal"><span>GOT SOMETHING IN MIND?</span><h2>Let’s find<br /><em>your thread.</em></h2><a className="landing-pill" href="/demo">Open the demo <Arrow /></a></div><span className="landing-outro-star" aria-hidden="true">✳</span></section>
    </main>
    <footer className="landing-footer"><a className="landing-logo" href="/" aria-label="Thread home">✳ THREAD</a><span>FASHION INTELLIGENCE / COSC2753 · 2026</span><button onClick={() => setMotion(value => !value)} aria-pressed={motion}>Motion {motion ? 'on' : 'off'} <span aria-hidden="true">{motion ? '◉' : '○'}</span></button><a href="#landing-main">Back to top ↑</a></footer>
  </div>;
}
