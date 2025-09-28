# frozen_string_literal: true

source "https://rubygems.org"

# Jekyll and plugins
gem "jekyll", "~> 4.3.0"
gem "minima", "~> 2.5"

# Jekyll plugins
gem "jekyll-feed", "~> 0.12"
gem "jekyll-sitemap", "~> 1.4"
gem "jekyll-seo-tag", "~> 2.8"

# GitHub Pages compatible theme
gem "github-pages", "~> 228", group: :jekyll_plugins

# Windows and JRuby compatibility
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]

# Add webrick for Ruby 3.0+
gem "webrick", "~> 1.7"